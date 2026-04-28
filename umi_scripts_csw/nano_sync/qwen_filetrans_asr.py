"""百炼 DashScope：qwen3-asr-flash-filetrans 异步转写；可选同步 ``qwen3-asr-flash`` 回退。

filetrans 需公网 ``file_url``。环境变量：``DASHSCOPE_API_KEY``。
"""

from __future__ import annotations

import base64
import json
import os
import time
import wave
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

_NS = Path(__file__).resolve().parent
_UMI_ROOT = _NS.parent

DASHSCOPE_API_BASE_CN = "https://dashscope.aliyuncs.com/api/v1"
DASHSCOPE_API_BASE_INTL = "https://dashscope-intl.aliyuncs.com/api/v1"
DEFAULT_FILETRANS_MODEL = "qwen3-asr-flash-filetrans"
DEFAULT_FLASH_SYNC_MODEL = "qwen3-asr-flash"
FLASH_GEN_PATH = "/services/aigc/multimodal-generation/generation"
# 官方文档：Base64 输入约 ≤10MB 编码后
_MAX_SYNC_AUDIO_BYTES = 9_500_000


def _load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip()
        if v.startswith('"') and v.endswith('"'):
            v = v[1:-1]
        if v.startswith("'") and v.endswith("'"):
            v = v[1:-1]
        if not k:
            continue
        if not os.environ.get(k):
            os.environ[k] = v


def load_env_around_script() -> None:
    for d in (_NS, _UMI_ROOT):
        _load_env_file(d / ".env")


def resolve_dashscope_key(explicit: str | None) -> str:
    if explicit and explicit.strip():
        return explicit.strip()
    v = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if v:
        return v
    raise SystemExit(
        "未设置 DASHSCOPE_API_KEY。请在 umi_scripts_csw/.env 或 nano_sync/.env 中写入一行：\n"
        "  DASHSCOPE_API_KEY=sk-你的百炼Key\n"
        "（勿提交到 git）"
    )


def _http_json(
    method: str,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any] | None = None,
    timeout: float = 120.0,
) -> dict[str, Any]:
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} {url}: {err_body[:4000]}") from e
    return json.loads(raw) if raw else {}


def _http_get_text(url: str, timeout: float = 120.0) -> str:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return resp.read().decode("utf-8")


def submit_filetrans_task(
    *,
    api_base: str,
    api_key: str,
    file_url: str,
    model: str = DEFAULT_FILETRANS_MODEL,
    language: str | None = "zh",
    enable_itn: bool = False,
    enable_words: bool = False,
    timeout: float = 120.0,
) -> str:
    base = api_base.rstrip("/")
    url = f"{base}/services/audio/asr/transcription"
    payload: dict[str, Any] = {
        "model": model,
        "input": {"file_url": file_url},
        "parameters": {
            "channel_id": [0],
            "enable_itn": enable_itn,
            "enable_words": enable_words,
        },
    }
    if language:
        payload["parameters"]["language"] = language
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-DashScope-Async": "enable",
    }
    j = _http_json("POST", url, headers, payload, timeout=timeout)
    out = j.get("output") or {}
    tid = out.get("task_id")
    if not tid:
        raise RuntimeError(f"提交转写任务失败，无 task_id: {json.dumps(j, ensure_ascii=False)[:2000]}")
    return str(tid)


def poll_filetrans_task(
    *,
    api_base: str,
    api_key: str,
    task_id: str,
    timeout_sec: float = 300.0,
    poll_interval_sec: float = 1.5,
) -> dict[str, Any]:
    base = api_base.rstrip("/")
    url = f"{base}/tasks/{task_id}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-DashScope-Async": "enable",
    }
    deadline = time.monotonic() + timeout_sec
    wait = poll_interval_sec
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = _http_json("GET", url, headers, None, timeout=120.0)
        out = last.get("output") or {}
        status = (out.get("task_status") or "").upper()
        if status == "SUCCEEDED":
            return last
        if status == "FAILED":
            msg = out.get("message") or out.get("code") or last
            raise RuntimeError(f"转写任务失败: {msg}")
        time.sleep(wait)
        wait = min(wait * 1.3, 5.0)
    raise TimeoutError(f"转写任务超时 ({timeout_sec}s): task_id={task_id} last={last!r}")


def fetch_transcription_document(task_json: dict[str, Any]) -> dict[str, Any]:
    out = task_json.get("output") or {}
    result = out.get("result") or {}
    turl = result.get("transcription_url")
    if turl:
        text = _http_get_text(str(turl), timeout=120.0)
        return json.loads(text)
    if "transcripts" in result:
        return result  # type: ignore[return-value]
    raise RuntimeError(
        "任务成功但无 transcription_url 且无内嵌 transcripts："
        + json.dumps(out, ensure_ascii=False)[:2000]
    )


def transcription_doc_to_whisper_shape(doc: dict[str, Any]) -> dict[str, Any]:
    """将 filetrans 结果 JSON 转为 ``pair_record_clips`` 可用的 ``verbose_json`` 形状。"""
    transcripts = doc.get("transcripts")
    if not isinstance(transcripts, list):
        transcripts = []

    rows: list[tuple[float, int, dict[str, Any]]] = []
    for ch_idx, tr in enumerate(transcripts):
        if not isinstance(tr, dict):
            continue
        for s in tr.get("sentences") or []:
            if not isinstance(s, dict):
                continue
            bt = float(int(s.get("begin_time", 0))) / 1000.0
            et = float(int(s.get("end_time", 0))) / 1000.0
            if et < bt:
                bt, et = et, bt
            tx = (s.get("text") or "").strip()
            rows.append(
                (
                    bt,
                    ch_idx,
                    {"start": bt, "end": et, "text": tx},
                )
            )
    rows.sort(key=lambda x: (x[0], x[1]))
    segments = [r[2] for r in rows]
    full_text = "".join(str(s.get("text", "")) for s in segments)
    duration = 0.0
    for s in segments:
        duration = max(duration, float(s.get("end", 0.0)))

    out: dict[str, Any] = {
        "text": full_text,
        "segments": segments,
        "duration": duration,
        "_source": "qwen3-asr-flash-filetrans",
    }
    tl = char_timeline_from_transcription_doc(doc)
    if tl is not None:
        out["_filetrans_char_timeline"] = tl
    return out


def char_timeline_from_transcription_doc(doc: dict[str, Any]) -> list[dict[str, float | str]] | None:
    """
    自 filetrans 原始 doc 提取字级时间轴（每句需含 ``words``，即须 ``enable_words``）。
    每个元素: ``t0``/``t1``(秒), ``s``(本格可见字串, 如单字+标点).
    若任一句无 words 则返回 None。
    """
    out: list[dict[str, float | str]] = []
    for tr in doc.get("transcripts") or []:
        if not isinstance(tr, dict):
            continue
        for s in tr.get("sentences") or []:
            if not isinstance(s, dict):
                continue
            wds = s.get("words")
            if not isinstance(wds, list) or not wds:
                return None
            for w in wds:
                if not isinstance(w, dict):
                    continue
                bt = float(int(w.get("begin_time", 0))) / 1000.0
                et = float(int(w.get("end_time", 0))) / 1000.0
                if et < bt:
                    bt, et = et, bt
                tpart = w.get("text") or ""
                punc = w.get("punctuation") or ""
                out.append({"t0": bt, "t1": et, "s": str(tpart) + str(punc)})
    return out or None


def _wav_file_duration_sec(wav: Path) -> float:
    with wave.open(str(wav), "rb") as w:
        return w.getnframes() / max(w.getframerate(), 1)


def run_flash_sync_wav_to_whisper_dict(
    *,
    api_key: str,
    wav_path: Path,
    model: str = DEFAULT_FLASH_SYNC_MODEL,
    language: str | None = "zh",
    enable_itn: bool = False,
    api_base: str = DASHSCOPE_API_BASE_CN,
    timeout: float = 180.0,
) -> dict[str, Any]:
    """同步 ``qwen3-asr-flash``（整段文本，无句级时间轴）。

    用于 filetrans 报 ``SUCCESS_WITH_NO_VALID_FRAGMENT`` 等、以拟声为主的短音频；
    将全文折成单条 ``segment`` 供二次推理使用。
    """
    raw = wav_path.read_bytes()
    if len(raw) > _MAX_SYNC_AUDIO_BYTES:
        raise SystemExit(
            f"WAV 过大 ({len(raw)} bytes)，同步 ASR 上限约 {_MAX_SYNC_AUDIO_BYTES}。"
            "请先用 ffmpeg 截短或降采样。"
        )
    dur = _wav_file_duration_sec(wav_path)
    suf = wav_path.suffix.lower()
    mime = "audio/wav" if suf in (".wav", ".wave") else "audio/mpeg"
    data_uri = f"data:{mime};base64,{base64.b64encode(raw).decode()}"
    body: dict[str, Any] = {
        "model": model,
        "input": {"messages": [{"role": "user", "content": [{"audio": data_uri}]}]},
        "parameters": {"asr_options": {"enable_itn": enable_itn}},
    }
    if language:
        body["parameters"]["asr_options"]["language"] = language
    url = api_base.rstrip("/") + FLASH_GEN_PATH
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    j = _http_json("POST", url, headers, body, timeout=timeout)
    if j.get("code"):
        raise RuntimeError(f"sync ASR: {j.get('code')} {j.get('message')}")
    out = j.get("output") or {}
    choices = out.get("choices") or []
    text_parts: list[str] = []
    for ch in choices:
        if not isinstance(ch, dict):
            continue
        msg = ch.get("message") or {}
        for c in msg.get("content") or []:
            if isinstance(c, dict) and c.get("text") is not None:
                text_parts.append(str(c.get("text", "")))
    full = "".join(text_parts).strip()
    t1 = max(float(dur), 1e-3)
    seg: list[dict[str, Any]] = [{"start": 0.0, "end": t1, "text": full or "(empty)"}]
    return {
        "text": full,
        "segments": seg,
        "duration": float(dur),
        "_source": "qwen3-asr-flash-sync-fallback",
    }


def run_filetrans_to_whisper_dict(
    *,
    api_base: str,
    api_key: str,
    file_url: str,
    model: str = DEFAULT_FILETRANS_MODEL,
    language: str | None = "zh",
    enable_itn: bool = False,
    enable_words: bool = False,
    timeout_sec: float = 300.0,
) -> dict[str, Any]:
    tid = submit_filetrans_task(
        api_base=api_base,
        api_key=api_key,
        file_url=file_url,
        model=model,
        language=language,
        enable_itn=enable_itn,
        enable_words=enable_words,
        timeout=120.0,
    )
    done = poll_filetrans_task(
        api_base=api_base,
        api_key=api_key,
        task_id=tid,
        timeout_sec=timeout_sec,
    )
    doc = fetch_transcription_document(done)
    return transcription_doc_to_whisper_shape(doc)
