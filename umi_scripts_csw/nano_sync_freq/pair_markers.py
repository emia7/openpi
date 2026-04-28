"""成对开始/结束时间戳，并输出 warnings。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class PairedClip:
    index: int
    t_start: float
    t_end: float
    t_stop_onset: float | None = None


@dataclass
class PairResult:
    clips: list[PairedClip]
    warnings: list[str] = field(default_factory=list)


def pair_start_stop_times(
    start_times: list[float],
    stop_times: list[float],
    *,
    duration_sec: float | None = None,
) -> PairResult:
    """
    贪心：每个 start 配**第一个严格在其之后**的 stop；剩余 stop 记 warning。
    """
    s = sorted(float(x) for x in start_times)
    t = sorted(float(x) for x in stop_times)
    w: list[str] = []
    clips: list[PairedClip] = []
    j = 0
    n = len(t)
    for ts in s:
        while j < n and t[j] <= ts + 1e-6:
            w.append(f"drop_stop_before_or_at_start: {t[j]:.3f} (start={ts:.3f})")
            j += 1
        if j >= n:
            w.append(f"unpaired_start: {ts:.3f}")
            continue
        te = t[j]
        j += 1
        if duration_sec is not None:
            if ts > float(duration_sec):
                w.append(f"start_after_effective_duration: {ts:.3f}")
                continue
            if te > float(duration_sec) + 1e-3:
                w.append(
                    f"clip_end_clamped: start={ts:.3f} stop {te:.3f} -> {float(duration_sec):.3f}"
                )
                te = float(duration_sec)
        if te > ts + 1e-6:
            clips.append(PairedClip(index=len(clips), t_start=ts, t_end=te))
    while j < n:
        w.append(f"unpaired_stop: {t[j]:.3f}")
        j += 1
    return PairResult(clips=clips, warnings=w)


def apply_stop_beep_tail(
    pr: PairResult,
    *,
    tail_sec: float,
    duration_sec: float | None = None,
    safety_before_next_start: float = 0.01,
) -> PairResult:
    """
    将每段 ``t_end`` 从「停标起音」延到起音+tail，且不超过下一段 ``t_start`` 与媒体末。
    """
    if tail_sec <= 0 or not pr.clips:
        return pr
    w = list(pr.warnings)
    out: list[PairedClip] = []
    n = len(pr.clips)
    for i, c in enumerate(pr.clips):
        onset = float(c.t_end)
        if c.t_stop_onset is not None:
            onset = float(c.t_stop_onset)
        te = onset + float(tail_sec)
        if i + 1 < n:
            te = min(te, pr.clips[i + 1].t_start - float(safety_before_next_start))
        if duration_sec is not None and float(duration_sec) > 0:
            te = min(te, float(duration_sec))
        if te < onset + tail_sec - 1e-4:
            w.append(
                f"clip_{c.index}_stop_tail_clamped:计划尾音 {tail_sec:.3f}s，"
                f"实际 {te - onset:.3f}s（下段开始或片末）"
            )
        out.append(
            PairedClip(
                index=c.index, t_start=c.t_start, t_end=te, t_stop_onset=onset
            )
        )
    return PairResult(clips=out, warnings=w)


def pair_to_jsonable(
    pr: PairResult, *, time_ref: Literal["marker_onset", "xcorr_argmax"] = "marker_onset"
) -> dict[str, Any]:
    return {
        "clips": [
            {
                "index": c.index,
                "t_start": c.t_start,
                "t_end": c.t_end,
                **({"t_stop_onset": c.t_stop_onset} if c.t_stop_onset is not None else {}),
            }
            for c in pr.clips
        ],
        "time_ref": time_ref,
        "warnings": pr.warnings,
    }
