"""
从 ASR 结果（含 ``segments`` 的 dict，如 Qwen filetrans 映射出的句级时间轴）中按「开始/停止录制」类锚点配对，得到**片段时间窗**（默认自「开始」句起点到「停止」句起点，见下）。

**默认**：每一轮为 [开始句起点, 停止句起点 + 可选口型余量)：`include_stop_mouth_sec` 可在「停」起音后多留一截便于看口型，不超过「停止」整句末。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

OrphanEnd = Literal["eof", "next_start", "none"]


def normalize_zh_text(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def matches_keywords(text: str, must_contain: tuple[str, ...] | list[str]) -> bool:
    n = normalize_zh_text(text)
    return all(k in n for k in must_contain)


@dataclass
class RecordClip:
    index: int
    t_start_content: float
    t_end_content: float
    t_marker_start_utter: float
    t_marker_start_utter_end: float
    t_marker_stop_utter: float
    t_marker_stop_utter_end: float
    warnings: list[str] = field(default_factory=list)


def _seg_bounds(raw: dict[str, Any]) -> tuple[float, float, str]:
    t0 = float(raw.get("start", 0.0))
    t1 = float(raw.get("end", 0.0))
    if t1 < t0:
        t0, t1 = t1, t0
    tx = (raw.get("text") or "").strip()
    return t0, t1, tx


def iter_sorted_segments(whisper_dict: dict[str, Any]) -> list[dict[str, Any]]:
    segs = whisper_dict.get("segments")
    if not segs or not isinstance(segs, list):
        return []
    return sorted(segs, key=lambda s: float(s.get("start", 0.0)))


def _per_char_from_timeline_rows(
    rows: list[dict[str, float | str]]
) -> list[tuple[str, float, float]]:
    """将字级行展开为 (字符, 起点秒, 终点秒)。一行内多字符时按等分插值。"""
    out: list[tuple[str, float, float]] = []
    for r in rows:
        seg = str(r.get("s", ""))
        t0, t1 = float(r.get("t0", 0.0)), float(r.get("t1", 0.0))
        if t1 < t0:
            t0, t1 = t1, t0
        n = max(len(seg), 1)
        for k, ch in enumerate(seg):
            t0i = t0 + (t1 - t0) * (k / n)
            t1i = t0 + (t1 - t0) * ((k + 1) / n)
            out.append((ch, t0i, t1i))
    return out


def _t0s_refine_if_leading_stretched(
    per_char: list[tuple[str, float, float]],
    a: int,
    end_idx: int,
    Lk: int,
    max_leading_char_sec: float = 0.9,
) -> float:
    """
    当口令**首字**在字级上被标得过长(常见于段尾、底噪/静音被误挂在「开」上)时，把
    **口令起剪**时刻改为后续第一个有时长的字的 ``t0``(跳过 0 长字)，使「开始录制」更贴近
    真实发声区。不用于「停止」：「停」的起音点通常应保留在首字 *t0*。
    """
    t0s = per_char[a][1]
    if end_idx - a < 2 or Lk < 1:
        return t0s
    w0 = per_char[a][2] - per_char[a][1]
    if w0 <= max_leading_char_sec:
        return t0s
    for j in range(1, end_idx - a):
        t0j, t1j = per_char[a + j][1], per_char[a + j][2]
        if t1j - t0j > 1e-3:
            return t0j
    return t0s


def _spans_phrase_from_per_char(
    per_char: list[tuple[str, float, float]],
    kw0: str,
    kw1: str,
    *,
    refine_t0s_long_lead: bool = False,
    max_leading_char_sec: float = 0.9,
) -> list[tuple[float, float]]:
    """
    在 per_char 拼接串上找 ``kw0+kw1``（及可选单字标点）每次出现的时间窗。
    用**固定字长**截断，避免 `re` 的 ``m.end()`` 在个别边界下跨度过大。
    """
    s = "".join(t[0] for t in per_char)
    nch = len(s)
    n_pc = len(per_char)
    if n_pc != nch:
        return []
    Lk = len(kw0) + len(kw1)
    PUNC = "。！？."
    segs: list[tuple[float, float]] = []
    a = 0
    while a <= nch - Lk:
        if s[a : a + Lk] != kw0 + kw1:
            a += 1
            continue
        end_idx = a + Lk
        if end_idx < nch and s[end_idx] in PUNC:
            end_idx += 1
        if end_idx <= a or a >= n_pc:
            a += 1
            continue
        t0s = per_char[a][1]
        if refine_t0s_long_lead:
            t0s = _t0s_refine_if_leading_stretched(
                per_char, a, end_idx, Lk, max_leading_char_sec
            )
        t1e = per_char[end_idx - 1][2]
        if t0s < t1e:
            segs.append((t0s, t1e))
        a = end_idx
    return segs


def marker_phrase_segments_from_char_timeline(
    whisper_dict: dict[str, Any],
    *,
    start_keywords: tuple[str, str] = ("开始", "录制"),
    stop_keywords: tuple[str, str] = ("停止", "录制"),
) -> list[dict[str, Any]] | None:
    """
    用 ``_filetrans_char_timeline``（需 filetrans 开启字级时间戳）在**全文**上
    匹配每轮「开始/停止」口令的**起止时间**，重排为交替的 ``segments``。
    失败时返回 None，回退到 :func:`expand_merged_start_stop_utterances`。
    """
    rows = whisper_dict.get("_filetrans_char_timeline")
    if not isinstance(rows, list) or not rows:
        return None
    per = _per_char_from_timeline_rows(rows)  # type: ignore[arg-type]
    if not per:
        return None
    starts = _spans_phrase_from_per_char(
        per,
        start_keywords[0],
        start_keywords[1],
        refine_t0s_long_lead=True,
    )
    stops = _spans_phrase_from_per_char(
        per, stop_keywords[0], stop_keywords[1]
    )
    if not starts or len(starts) != len(stops):
        return None
    n = len(starts)
    for i in range(n):
        sa, sb = starts[i]
        ta, tb = stops[i]
        if not (sa < sb and ta < tb and sb <= ta):
            return None
        if i < n - 1 and starts[i + 1][0] < tb - 1e-4:
            return None
    punc = "。"
    segs: list[dict[str, Any]] = []
    for i in range(n):
        s0, s1 = starts[i]
        t0, t1 = stops[i]
        segs.append(
            {
                "start": s0,
                "end": s1,
                "text": f"{start_keywords[0]}{start_keywords[1]}{punc}",
            }
        )
        segs.append(
            {
                "start": t0,
                "end": t1,
                "text": f"{stop_keywords[0]}{stop_keywords[1]}{punc}",
            }
        )
    return segs


def expand_merged_start_stop_utterances(
    segments: list[dict[str, Any]],
    *,
    start_keywords: tuple[str, str] = ("开始", "录制"),
    stop_keywords: tuple[str, str] = ("停止", "录制"),
    marker_phrase_time_frac: float = 0.15,
) -> list[dict[str, Any]]:
    """
    当 VAD/断句把多轮「开始/停止」并成单条 ``segment`` 时，按**轮次**在整段
    时间窗内拆出多对「开始句 / 停止句」时间戳，再交给 :func:`pair_record_clips`。
    每轮内为口令分配 ``marker_phrase_time_frac`` 时长的边界（启发式，仅用于合并段）。
    """
    out: list[dict[str, Any]] = []
    frac = float(marker_phrase_time_frac)
    if not (0.0 < frac < 0.5):
        frac = 0.15
    for raw in sorted(segments, key=lambda s: float(s.get("start", 0.0))):
        t0, t1, text = _seg_bounds(raw)
        is_start = matches_keywords(text, start_keywords)
        is_stop = matches_keywords(text, stop_keywords)
        if not (is_start and is_stop):
            out.append(raw)
            continue
        n0 = len(re.findall("开始", normalize_zh_text(text)))
        n1 = len(re.findall("停止", normalize_zh_text(text)))
        if n0 < 1 or n0 != n1:
            out.append(raw)
            continue
        k = n0
        span = t1 - t0
        cycle = span / max(k, 1)
        wmark = min(frac * cycle, cycle / 2.0 - 1e-4)  # 保中间内容为正
        for j in range(k):
            c0 = t0 + j * cycle
            c1 = t0 + (j + 1) * cycle
            a0, a1 = c0, c0 + wmark
            b0, b1 = c1 - wmark, c1
            if a1 > b0:
                mid = 0.5 * (c0 + c1)
                a0, a1 = c0, max(c0, mid - 0.1)
                b0, b1 = min(c1, mid + 0.1), c1
            out.append(
                {
                    "start": a0,
                    "end": a1,
                    "text": "开始录制" + ("。" if "。" in text else "."),
                }
            )
            out.append(
                {
                    "start": b0,
                    "end": b1,
                    "text": "停止录制" + ("。" if "。" in text else "."),
                }
            )
    return out


def pair_record_clips(
    whisper_dict: dict[str, Any],
    duration_sec: float,
    *,
    start_keywords: tuple[str, str] = ("开始", "录制"),
    stop_keywords: tuple[str, str] = ("停止", "录制"),
    after_start_sec: float = 0.0,
    before_stop_sec: float = 0.0,
    include_stop_mouth_sec: float = 0.0,
    orphan_end: OrphanEnd = "eof",
) -> tuple[list[RecordClip], list[str]]:
    process_warnings: list[str] = []
    segs = iter_sorted_segments(whisper_dict)
    clips: list[RecordClip] = []
    pending: tuple[float, float] | None = None
    clip_index = 0

    for raw in segs:
        t0, t1, text = _seg_bounds(raw)
        is_start = matches_keywords(text, start_keywords)
        is_stop = matches_keywords(text, stop_keywords)
        if is_start and is_stop:
            process_warnings.append(f"同一段同时命中 start/stop，已忽略: {text!r} @ {t0:.3f}")
            continue
        if is_start:
            if pending is not None:
                process_warnings.append("duplicate_start_before_stop: 以新的「开始」为准，上一段无「停止」")
            pending = (t0, t1)
            continue
        if is_stop:
            if pending is None:
                process_warnings.append("orphan_stop: 有「停止」但前面没有「开始」")
                continue
            s0, s1 = pending[0], pending[1]
            # 片段时间窗：从「开始」句起点，到「停」起音后含一小段口型(不超过整句「停止」末、不超过媒体末)
            t_content0 = s0 + after_start_sec
            t_proposed = t0 - before_stop_sec + include_stop_mouth_sec
            t_content1 = min(
                t_proposed,
                float(t1),
                float(duration_sec),
            )
            cw: list[str] = []
            if include_stop_mouth_sec > 0.0 and t_proposed > float(t1) + 1e-6:
                cw.append("include_stop_mouth_clamped_to_stop_utterance_end")
            if t_content1 < t_content0:
                cw.append("nonpositive_content_len_clamped")
                t_content1 = t_content0
            clips.append(
                RecordClip(
                    index=clip_index,
                    t_start_content=t_content0,
                    t_end_content=t_content1,
                    t_marker_start_utter=s0,
                    t_marker_start_utter_end=s1,
                    t_marker_stop_utter=t0,
                    t_marker_stop_utter_end=t1,
                    warnings=cw,
                )
            )
            clip_index += 1
            pending = None
            continue

    if pending is not None:
        s0, s1 = pending[0], pending[1]
        if orphan_end == "eof":
            t_content0 = s0 + after_start_sec
            t_content1 = max(duration_sec, 0.0) - 0.0
            if t_content1 < t_content0:
                t_content1 = t_content0
            process_warnings.append("unclosed_start: 无「停止」，按 orphan_end=eof 延伸到文件末")
            clips.append(
                RecordClip(
                    index=clip_index,
                    t_start_content=t_content0,
                    t_end_content=t_content1,
                    t_marker_start_utter=s0,
                    t_marker_start_utter_end=s1,
                    t_marker_stop_utter=-1.0,
                    t_marker_stop_utter_end=-1.0,
                    warnings=["unclosed_start"],
                )
            )
        elif orphan_end == "next_start":
            process_warnings.append("unclosed_start: orphan_end=next_start 未实现，请用 eof 或 none")
        else:
            process_warnings.append("unclosed_start: orphan_end=none，已丢弃无停止的待配对区段")

    return clips, process_warnings


def record_clips_to_jsonable(clips: list[RecordClip]) -> list[dict[str, Any]]:
    return [
        {
            "index": c.index,
            "t_start_content": c.t_start_content,
            "t_end_content": c.t_end_content,
            "t_marker_start_utter": c.t_marker_start_utter,
            "t_marker_start_utter_end": c.t_marker_start_utter_end,
            "t_marker_stop_utter": c.t_marker_stop_utter,
            "t_marker_stop_utter_end": c.t_marker_stop_utter_end,
            "warnings": c.warnings,
        }
        for c in clips
    ]
