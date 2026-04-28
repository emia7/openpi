#!/usr/bin/env python3
"""
用百炼 Qwen filetrans（句级时间戳）转写，按「开始录制/停止录制」式锚点输出片段时间窗；
可选对视频/音轨用 ffmpeg 切片。

ASR 需公网可访问的音频 URL（``--file-url``），与本地 ``path`` 应对应同一段音画（通常先将 ``path``
中的音轨导出上传后，把 HTTPS 链接填入 ``--file-url``）。

若 ASR 结果无 segments，会报错退出（无法作时间切分）。

用法（在 umi_scripts_csw 下）:
  python nano_sync/segment_by_record_markers.py video.mp4 --file-url https://.../a.wav --out-json out.json
  python nano_sync/segment_by_record_markers.py video.mp4 --file-url https://.../a.wav --cut-dir ./cuts

默认每段 MP4 为 [「开始录制」句起点, 「停止」起音后带一小段口型)（不超过「停止」整句末、媒体末）；
片头带「开始」口令声；``--include-stop-mouth-sec`` 控制口型余量，见 ``pair_record_clips``。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import wave
from pathlib import Path
from typing import Any

_NS = Path(__file__).resolve().parent
_UMI = _NS.parent
for d in (_NS, _UMI):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

from audio_extract import find_ffmpeg, probe_audio_with_ffmpeg  # noqa: E402
from record_marker_phrases import (  # noqa: E402
    DEFAULT_START_KEYWORDS,
    DEFAULT_STOP_KEYWORDS,
    DEFAULT_START_PHRASE,
    DEFAULT_STOP_PHRASE,
)
from record_marker_segments import (  # noqa: E402
    expand_merged_start_stop_utterances,
    marker_phrase_segments_from_char_timeline,
    pair_record_clips,
    record_clips_to_jsonable,
)
from qwen_filetrans_asr import (  # noqa: E402
    DASHSCOPE_API_BASE_CN,
    DEFAULT_FILETRANS_MODEL,
    load_env_around_script,
    resolve_dashscope_key,
    run_filetrans_to_whisper_dict,
)


def _wav_duration_sec(wav: Path) -> float:
    with wave.open(str(wav), "rb") as w:
        return w.getnframes() / max(w.getframerate(), 1)


def _ffmpeg_has_audio(path: Path) -> bool:
    if shutil.which("ffmpeg") is None:
        return False
    p = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "default=nk=1:nw=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if p.returncode != 0:
        return False
    return "audio" in (p.stdout or "").lower()


def cut_ffmpeg(
    in_media: Path,
    out_path: Path,
    t0: float,
    t1: float,
) -> None:
    try:
        ff = find_ffmpeg()
    except FileNotFoundError as e:
        raise RuntimeError("未找到 ffmpeg") from e
    out_path.parent.mkdir(parents=True, exist_ok=True)
    d = t1 - t0
    if d <= 0:
        raise ValueError("切片时长须为正")
    cmd = [
        ff,
        "-y",
        "-ss",
        f"{t0:.3f}",
        "-i",
        str(in_media),
        "-t",
        f"{d:.3f}",
        "-c",
        "copy",
        str(out_path),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if p.returncode != 0:
        raise RuntimeError(
            f"ffmpeg 失败: {p.stderr[-2000:] if p.stderr else p.stdout or 'unknown'}"
        )


def _parse_pair_keyword(s: str) -> tuple[str, str]:
    s = s.strip()
    for sep in ("+", ",", "，", " "):
        if sep in s:
            a, _, b = s.partition(sep)
            a, b = a.strip(), b.strip()
            if a and b:
                return a, b
    raise SystemExit(
        f"关键词需两项，用 +/逗号/空格分隔: {s!r}"
    )


def _ffprobe_duration_sec(m: Path) -> float:
    p = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(m),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if p.returncode != 0 or not (p.stdout or "").strip():
        return 0.0
    try:
        return max(0.0, float(p.stdout.strip()))
    except ValueError:
        return 0.0


def _resolve_duration_for_tr(path: Path, tr: dict[str, Any], tmp_wav: Path | None) -> float:
    d = float(tr.get("duration", 0.0) or 0.0)
    if d > 0:
        return d
    if tmp_wav and tmp_wav.is_file():
        return _wav_duration_sec(tmp_wav)
    if path.suffix.lower() in (".wav", ".wave"):
        return _wav_duration_sec(path)
    return _ffprobe_duration_sec(path)


def main() -> None:
    load_env_around_script()
    p = argparse.ArgumentParser(description="Qwen filetrans ASR + 起止录制口令 分段")
    p.add_argument("path", type=Path, help="本地视频、音频或 WAV（切片与时长；ASR 用 --file-url）")
    p.add_argument(
        "--file-url",
        default=None,
        help="公网 HTTPS 可访问的音频 URL（与 path 音轨应对齐）。调用 API 时必填（除非 --from-json）",
    )
    p.add_argument(
        "--asr-model",
        default=DEFAULT_FILETRANS_MODEL,
        help="DashScope 录音文件识别模型，默认 qwen3-asr-flash-filetrans",
    )
    p.add_argument(
        "--dashscope-api-base",
        default=os.environ.get("DASHSCOPE_API_BASE", DASHSCOPE_API_BASE_CN),
        help="DashScope REST 根路径，默认北京；国际可用 DASHSCOPE_API_BASE 或本参数",
    )
    p.add_argument(
        "--dashscope-api-key",
        default=None,
        help="覆盖环境变量 DASHSCOPE_API_KEY",
    )
    p.add_argument("--timeout", type=float, default=300.0, help="异步转写轮询总超时(秒)")
    p.add_argument(
        "--language",
        default="zh",
        help="识别语种，如 zh；不确定可传空字符串以省略 language 参数",
    )
    p.add_argument(
        "--enable-itn",
        action="store_true",
        help="开启中文/英文逆文本标准化（默认关闭）",
    )
    p.add_argument(
        "--enable-words",
        action="store_true",
        help="字级时间戳(默认关)。**建议开启**: 多轮「开始/停止」被 VAD 并成一句时，可配合 ``_filetrans_char_timeline`` 用字级时间对齐，否则对合并段只能用启发式比例拆分",
    )
    p.add_argument(
        "--start-keywords",
        type=str,
        default=None,
        help=f"两项，逗号/空格/加号分隔。默认: {'+'.join(DEFAULT_START_KEYWORDS)}",
    )
    p.add_argument(
        "--stop-keywords",
        type=str,
        default=None,
        help=f"默认: {'+'.join(DEFAULT_STOP_KEYWORDS)}",
    )
    p.add_argument(
        "--content-after-start-sec",
        type=float,
        default=0.0,
        help="在「开始录制」句起点之后再推迟开剪(秒)；默认0即从「开始」发声起剪",
    )
    p.add_argument(
        "--before-stop-sec",
        type=float,
        default=0.0,
        help="在「停止录制」句起点之前再提前结束(秒)；默认0",
    )
    p.add_argument(
        "--include-stop-mouth-sec",
        type=float,
        default=0.12,
        help="从「停止」句起点起再向后多留几秒便于看口型；与 before-stop 叠加，且不超过该句末、媒体末。传0可关闭",
    )
    p.add_argument(
        "--orphan-end",
        choices=["eof", "next_start", "none"],
        default="eof",
        help="有开始无停止时: 延伸到文件末 / 未实现 / 丢弃",
    )
    p.add_argument("--out-json", type=Path, default=None, help="写 JSON 文件；默认 stdout")
    p.add_argument(
        "--cut-dir",
        type=Path,
        default=None,
        help="将每段内容区裁成新文件(依赖 ffmpeg, stream copy；视频需含音轨时亦 copy)",
    )
    p.add_argument(
        "--asr-only", action="store_true", help="只打印 ASR 规整后的 JSON（含 segments，调试用）"
    )
    p.add_argument(
        "--from-json", type=Path, default=None, help="不调用 API,直接读已保存的 verbose_json 文件"
    )
    args = p.parse_args()
    path = args.path.expanduser()
    if not path.is_file():
        raise SystemExit(f"file not found: {path}")

    skw = _parse_pair_keyword(
        args.start_keywords or f"{DEFAULT_START_KEYWORDS[0]}+{DEFAULT_START_KEYWORDS[1]}"
    )
    stw = _parse_pair_keyword(
        args.stop_keywords or f"{DEFAULT_STOP_KEYWORDS[0]}+{DEFAULT_STOP_KEYWORDS[1]}"
    )
    tr: dict[str, Any] | None = None
    tmp_wav: Path | None = None

    if args.from_json:
        with open(args.from_json, encoding="utf-8") as f:
            tr = json.load(f)
    else:
        file_url = (args.file_url or "").strip()
        if not file_url:
            raise SystemExit(
                "调用百炼 ASR 时必须提供 --file-url（公网可访问的音频 HTTPS 链接，内容需与 path 音轨一致）。\n"
                "若仅调试分段逻辑，请使用 --from-json 指向已保存的 JSON。\n"
                "Key 写在 umi_scripts_csw/.env：DASHSCOPE_API_KEY=sk-..."
            )

        if path.suffix.lower() in (".mp4", ".mov", ".m4v", ".webm", ".mkv"):
            prob = probe_audio_with_ffmpeg(path)
            if not prob.get("has_audio"):
                raise SystemExit("无音轨")
            if not _ffmpeg_has_audio(path):
                print("警告: ffprobe 未确认音轨,仍继续", file=sys.stderr)
        elif path.suffix.lower() in (".wav", ".wave"):
            tmp_wav = path
        else:
            raise SystemExit("不支持的输入格式，请用 mp4/mov/webm 或 wav")

        key = resolve_dashscope_key(args.dashscope_api_key)
        api_base = str(args.dashscope_api_base).rstrip("/")
        lang = (args.language or "").strip() or None
        print(
            f"[api] ASR model={args.asr_model!r} base={api_base!r} file_url={file_url[:80]!r}… "
            f"(asr-only={args.asr_only})",
            flush=True,
        )
        tr = run_filetrans_to_whisper_dict(
            api_base=api_base,
            api_key=key,
            file_url=file_url,
            model=args.asr_model,
            language=lang,
            enable_itn=bool(args.enable_itn),
            enable_words=bool(args.enable_words),
            timeout_sec=float(args.timeout),
        )
        if args.asr_only:
            out = json.dumps(tr, ensure_ascii=False, default=str, indent=2)
            if args.out_json:
                args.out_json.write_text(out, encoding="utf-8")
                print(f"已写: {args.out_json}", flush=True)
            else:
                print(out, flush=True)
            return

    assert tr is not None
    if not (tr.get("segments") and len(tr["segments"]) > 0):
        raise SystemExit(
            "ASR 结果无分段时间轴(segments 为空)，无法作锚点切分。"
        )
    segs0 = tr.get("segments")
    if isinstance(segs0, list) and segs0:
        alt = marker_phrase_segments_from_char_timeline(
            tr, start_keywords=skw, stop_keywords=stw
        )
        if alt is not None:
            tr = {**tr, "segments": alt}
        else:
            tr = {
                **tr,
                "segments": expand_merged_start_stop_utterances(
                    segs0, start_keywords=skw, stop_keywords=stw
                ),
            }
    duration_sec = _resolve_duration_for_tr(path, tr, tmp_wav)
    if duration_sec <= 0 and tmp_wav and tmp_wav.suffix.lower() in (".wav", ".wave"):
        duration_sec = _wav_duration_sec(tmp_wav)
    if duration_sec <= 0:
        duration_sec = _ffprobe_duration_sec(path)
    clips, pw = pair_record_clips(
        tr,
        duration_sec,
        start_keywords=skw,
        stop_keywords=stw,
        after_start_sec=args.content_after_start_sec,
        before_stop_sec=args.before_stop_sec,
        include_stop_mouth_sec=args.include_stop_mouth_sec,
        orphan_end=args.orphan_end,  # type: ignore[arg-type]
    )
    payload: dict[str, Any] = {
        "file": str(path) if path.is_file() else None,
        "duration_sec": duration_sec,
        "marker_defaults": {
            "start_phrase": DEFAULT_START_PHRASE,
            "stop_phrase": DEFAULT_STOP_PHRASE,
            "start_keywords": list(skw),
            "stop_keywords": list(stw),
            "include_stop_mouth_sec": float(args.include_stop_mouth_sec),
        },
        "asr_excerpt": {
            "text": (tr.get("text") or "")[:2000],
            "segment_count": len(tr.get("segments") or []),
        },
        "process_warnings": pw,
        "clips": record_clips_to_jsonable(clips),
    }
    if args.cut_dir:
        cut_root = args.cut_dir.expanduser()
        for c in clips:
            ext = path.suffix.lower() if path.suffix else ".mp4"
            if ext not in (".mp4", ".mov", ".webm", ".mkv", ".wav", ".m4a"):
                ext = ".mp4"
            oname = f"clip_{c.index:04d}{ext}"
            if not path.is_file():
                raise SystemExit("切片需要可读的原始媒体 path")
            cut_ffmpeg(
                path,
                cut_root / oname,
                c.t_start_content,
                c.t_end_content,
            )
        print(f"[ffmpeg] 已写入 {len(clips)} 个文件到 {cut_root}", flush=True)

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out_json:
        args.out_json.write_text(text, encoding="utf-8")
    else:
        print(text, flush=True)


if __name__ == "__main__":
    main()
