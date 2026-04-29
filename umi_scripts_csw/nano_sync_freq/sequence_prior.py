"""
时间轴上 S/T 事件流的先验：默认可为「起-停-起-停…」；在「期待下一段起」时若先出现多馀「停」，
可丢弃至多 max 个误触；超过则标人工。若「期待本段停」时先出现「起」，标人工（可能缺省上一停）。
"""

from __future__ import annotations

from typing import Any

from . import config


def apply_st_alternation_prior(
    merged_events: list[dict[str, Any]],
    sample_rate: int,
) -> tuple[list[int], list[int], list[str], list[dict[str, Any]]]:
    """
    ``merged_events``：按时间排序，每项含 ``k``（样本下标）、``label``（``\"S\"``/``\"T\"``）；
    可选 ``manual_ambig`` 为真时仍参与状态机，但会写入 ``manual_review`` 供人复核。

    返回：``(p_s, p_t, log_lines, manual_review)``，下标与输入峰一致，仅可能**删**多馀 T。
    """
    if not merged_events:
        return [], [], [], []
    evs = sorted(merged_events, key=lambda e: int(e["k"]))
    max_sp = int(
        max(0, getattr(config, "MAX_SPURIOUS_T_BEFORE_NEXT_S", 3) or 0)
    )
    out_s: list[int] = []
    out_t: list[int] = []
    log: list[str] = []
    manual: list[dict[str, Any]] = []
    expect: str = "S"
    spurious_t_run = 0
    t_sec = lambda kk: float(kk) / float(sample_rate)

    for e in evs:
        k = int(e["k"])
        lab = str(e.get("label", ""))
        if lab not in ("S", "T"):
            continue
        if e.get("manual_ambig"):
            manual.append(
                {
                    "k": k,
                    "t": t_sec(k),
                    "reason": "low_ncc_margin",
                    "label": lab,
                }
            )
        if expect == "S":
            if lab == "S":
                out_s.append(k)
                expect = "T"
                spurious_t_run = 0
            else:
                spurious_t_run += 1
                if spurious_t_run <= max_sp:
                    log.append(
                        f"prior_drop_t: t={t_sec(k):.4f}s (spurious before next S) #{spurious_t_run}"
                    )
                else:
                    manual.append(
                        {
                            "k": k,
                            "t": t_sec(k),
                            "reason": "excess_t_before_s",
                            "label": "T",
                        }
                    )
                    spurious_t_run = 0
                    log.append(
                        f"prior_manual: excess T before next S t={t_sec(k):.4f}s (>{max_sp} dropped already since last S)"
                    )
        else:
            if lab == "T":
                out_t.append(k)
                expect = "S"
                spurious_t_run = 0
            else:
                manual.append(
                    {
                        "k": k,
                        "t": t_sec(k),
                        "reason": "s_without_prior_stop",
                        "label": "S",
                    }
                )
                out_s.append(k)
                expect = "T"
                spurious_t_run = 0
                log.append(f"prior_manual: early S t={t_sec(k):.4f}s (expected T)")

    return sorted(out_s), sorted(out_t), log, manual
