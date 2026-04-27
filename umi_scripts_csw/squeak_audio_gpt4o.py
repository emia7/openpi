#!/usr/bin/env python3
"""
VectorEngine / 任意 OpenAI 兼容中转：用 **GPT-4o 系** 模型听整段音轨，识别
「短促尖声/尖叫鸡类」事件并给出大约时刻（按提示输出 JSON）。

**平台与模型选择（与 Qwen 无关）**
- 在 [VectorEngine](https://vectorengine.ai) 这类 **OpenAI 协议** 中台上，优先选 **`gpt-4o`**
  作为多模态旗舰；若你账号下该 ID 对 **input_audio** 报不支持，可改控制台里
  列出的同族名，常见备选：**`gpt-4o-audio-preview`**、**`gpt-4o-2024-11-20`**
  等，**以你后台「模型 ID」为准**。
- 定价与倍率见: https://vectorengine.ai/pricing

**调用形态**（与 OpenAI Chat Completions 一致，见 Simon Willison 对 audio input 的说明）：
- `messages[0].content` 为数组：`input_audio` 里 **`data` 为裸 Base64**（不要带
  `data:audio/...;base64,` 前缀），`format` 为 `wav` 或 `mp3`。
- 仅要文字时设 `modalities: ["text"]`，不请求语音合成。

**环境**
- `OPENAI_API_KEY` 或 `VECTOR_ENGINE_API_KEY`（与下面解析顺序一致）
- `OPENAI_BASE_URL` 可代替 `--base-url`，便于与 openpi 里其它客户端共用配置
- 若存在与本脚本同目录（或 openpi 根目录）的 **`.env` 文件**（`KEY=值` 每行），
  启动时自动读入；**不要**把 Key 提交到 git。示例见同目录 `squeak_audio_gpt4o.env.example`.

**Token / 费用**：以响应中的 **`[usage]`** 为准（`total_tokens` 等）。不同中转对
同模型的计费倍率见 [VectorEngine 定价](https://vectorengine.ai/pricing)（页面为参考，
扣费以控制台为准）。

依赖: pip install "openai>=1.52"
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

_UMI = Path(__file__).resolve().parent
if str(_UMI) not in sys.path:
    sys.path.insert(0, str(_UMI))
from audio_extract import extract_audio_wav, load_wav_f32, probe_audio_with_ffmpeg


DEFAULT_PROMPT = """你是音频分析助手。请听这段音轨，判断：
1) 是否有「短促、偏尖锐」的拟声/玩具尖叫类声音（如橡皮鸭、尖叫鸡），与说话声、环境噪音区分。
2) 若有，按时间顺序列出每一次的大约发生时刻（秒，相对本段从 0s 起，可保留一位小数）。
3) 没有则明确说明。

只输出一个 JSON 对象，不要其它文字，格式如下：
{"has_squeak": true或false, "events": [{"t_peak_sec": 数}], "note": "一句可选说明" }
"""


def _load_env_file(path: Path) -> None:
    """不依赖 python-dotenv：从 .env 读入，已有同名环境变量则不覆盖。"""
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


def _load_env_around_script() -> None:
    for d in (_UMI, _UMI.parent):
        _load_env_file(d / ".env")


def _resolve_api_key(explicit: str | None) -> str:
    if explicit and explicit.strip():
        return explicit.strip()
    for name in ("OPENAI_API_KEY", "VECTOR_ENGINE_API_KEY", "VE_API_KEY"):
        v = os.environ.get(name, "").strip()
        if v:
            return v
    raise SystemExit(
        "未设置 API Key。请设 OPENAI_API_KEY 或 VECTOR_ENGINE_API_KEY，或传 --api-key"
    )


def _wav_b64(wav_path: Path) -> str:
    return base64.b64encode(wav_path.read_bytes()).decode("ascii")


def run_gpt4o_audio(
    *,
    wav: Path,
    model: str,
    prompt: str,
    base_url: str,
    api_key: str,
    stream: bool,
    use_modalities_text: bool = True,
    timeout_s: float = 300.0,
) -> tuple[str, Any]:
    try:
        from openai import OpenAI
    except ImportError as e:
        raise SystemExit('请安装: python3 -m pip install "openai>=1.52"（若本机无 pip 命令请用此写法）') from e

    b64 = _wav_b64(wav)
    # 带整段音频时请求体大，转盘中继可能 30s～数分钟才有首包，勿误以为卡死
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s)
    # OpenAI 标准：仅文本模态，避免 TTS
    user_content: list[dict] = [
        {"type": "text", "text": prompt},
        {
            "type": "input_audio",
            "input_audio": {"data": b64, "format": "wav"},
        },
    ]
    extra: dict = {}
    if use_modalities_text:
        extra["modalities"] = ["text"]
    if stream:
        comp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": user_content}],
            stream=True,
            stream_options={"include_usage": True},
            **extra,
        )
        acc: list[str] = []
        usage: Any = None
        for ch in comp:
            if ch.choices and ch.choices[0].delta.content:
                acc.append(ch.choices[0].delta.content)
            if getattr(ch, "usage", None) is not None:
                usage = ch.usage
        return "".join(acc), usage
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": user_content}],
        stream=False,
        **extra,
    )
    text = (r.choices[0].message.content or "").strip()
    return text, getattr(r, "usage", None)


def _print_usage(usage: Any) -> None:
    if usage is None:
        print("[usage] (网关未返回 usage 字段，无法从本机得知 token 明细)", flush=True)
        return
    try:
        d = usage.model_dump() if hasattr(usage, "model_dump") else dict(usage)
    except Exception:
        d = str(usage)
    print(f"[usage] {json.dumps(d, ensure_ascii=False, default=str)}", flush=True)
    if isinstance(d, dict) and d.get("total_tokens") is not None:
        print(
            f"[usage] 合计 total_tokens={d.get('total_tokens')} "
            f"(prompt={d.get('prompt_tokens', '?')} + completion={d.get('completion_tokens', '?')})",
            flush=True,
        )


def main() -> None:
    _load_env_around_script()
    p = argparse.ArgumentParser(
        description="VectorEngine/OpenAI 兼容：GPT-4o 听整段 WAV 或从视频抽轨，标尖叫类时刻"
    )
    p.add_argument("path", type=Path, help="视频 MP4 等 或 已有 WAV")
    p.add_argument(
        "--model",
        default="gpt-4o",
        help="默认 gpt-4o；听轨若报模型不支持，可改 gpt-4o-audio-preview 等(见控制台)",
    )
    p.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", "https://api.vectorengine.ai/v1"),
        help="默认先读环境变量 OPENAI_BASE_URL，否则为 VectorEngine 常见 /v1",
    )
    p.add_argument("--api-key", default=None, help="不读环境变量时显式传 Key")
    p.add_argument(
        "--no-modalities",
        action="store_true",
        help="不传 modalities(部分兼容网关对纯文本不识别该字段时可试)",
    )
    p.add_argument("--stream", action="store_true", help="流式输出")
    p.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="HTTP 总超时(秒)。整段音频+中转可能很慢，默认 300",
    )
    p.add_argument("--keep-wav", type=Path, default=None, help="从视频抽轨时另存 WAV 到该路径")
    p.add_argument("--prompt-file", type=Path, default=None, help="自定义系统提示(覆盖默认 JSON 任务)")
    args = p.parse_args()
    path = args.path.expanduser()
    if not path.is_file():
        raise SystemExit(f"文件不存在: {path}")

    tmp_wav: Path | None = None
    if path.suffix.lower() in (".mp4", ".mov", ".m4v", ".webm", ".mkv"):
        prob = probe_audio_with_ffmpeg(path)
        if not prob.get("has_audio"):
            raise SystemExit("无音轨")
        d = Path(tempfile.mkdtemp(prefix="gpt4o_audio_"))
        tmp_wav = d / f"{path.stem}.wav"
        extract_audio_wav(path, tmp_wav, sample_rate=44100, mono=True)
        x, sr = load_wav_f32(tmp_wav)
        print(f"[local] 已抽轨 sr={sr} 时长={x.size / max(sr, 1):.3f}s  tmp={tmp_wav}", flush=True)
        if args.keep_wav:
            k = args.keep_wav.expanduser()
            k.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(tmp_wav, k)
            print(f"[local] 已另存: {k}", flush=True)
        wav = tmp_wav
    else:
        wav = path

    prompt = args.prompt_file.read_text(encoding="utf-8") if args.prompt_file else DEFAULT_PROMPT
    key = _resolve_api_key(args.api_key)

    use_mod = not args.no_modalities
    print(
        f"[api] model={args.model!r}  base_url={args.base_url!r}  modalities_text={use_mod!r}  timeout={args.timeout:.0f}s",
        flush=True,
    )
    print(
        "[api] 正在请求云端（整段 WAV→Base64，体量大，中转常需 30s～数分钟，请等这一行之后的结果）",
        flush=True,
    )
    try:
        text, usage = run_gpt4o_audio(
            wav=wav,
            model=args.model,
            prompt=prompt,
            base_url=args.base_url.rstrip("/"),
            api_key=key,
            stream=args.stream,
            use_modalities_text=use_mod,
            timeout_s=args.timeout,
        )
    except Exception as e:
        print(f"\n[api] 请求失败: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        if "modalities" in str(e).lower() or "400" in str(e):
            print(
                "提示: 可再试: --no-modalities 或把 --model 换为 gpt-4o-audio-preview",
                file=sys.stderr,
                flush=True,
            )
        raise
    _print_usage(usage)
    print("\n--- 输出 ---\n", flush=True)
    print(text, flush=True)

    if tmp_wav and tmp_wav.parent.exists():
        try:
            shutil.rmtree(tmp_wav.parent, ignore_errors=True)
        except OSError:
            pass


if __name__ == "__main__":
    main()
