"""D. 成对纯逻辑单测。"""

from __future__ import annotations

from nano_sync_freq.pair_markers import apply_stop_beep_tail, pair_start_stop_times


def test_apply_stop_tail() -> None:
    pr = pair_start_stop_times([0.0, 5.0], [2.0, 8.0], duration_sec=20.0)
    pr2 = apply_stop_beep_tail(pr, tail_sec=0.1, duration_sec=20.0)
    assert pr2.clips[0].t_end == 2.0 + 0.1
    assert pr2.clips[0].t_stop_onset == 2.0
    assert pr2.clips[1].t_start == 5.0
    assert pr2.clips[1].t_end == 8.0 + 0.1
    pr = pair_start_stop_times([0.0, 3.0], [1.0, 4.0], duration_sec=10.0)
    assert len(pr.clips) == 2
    assert pr.clips[0].t_start == 0.0 and pr.clips[0].t_end == 1.0
    assert pr.clips[1].t_start == 3.0 and pr.clips[1].t_end == 4.0
    assert not pr.warnings


def test_unpaired_trailing_stop() -> None:
    pr = pair_start_stop_times([1.0], [2.0, 3.0], duration_sec=10.0)
    assert len(pr.clips) == 1
    assert any("unpaired_stop" in w for w in pr.warnings)


def test_unpaired_start() -> None:
    pr = pair_start_stop_times([1.0, 2.0], [3.0], duration_sec=10.0)
    assert any("unpaired" in w for w in pr.warnings)
