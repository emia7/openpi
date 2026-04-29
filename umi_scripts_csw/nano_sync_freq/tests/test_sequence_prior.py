"""S/T 交替先验单测（不跑整段 NCC）。"""

from __future__ import annotations

from nano_sync_freq import sequence_prior


def test_prior_sts_perfect() -> None:
    ev = [
        {"k": 0, "label": "S", "manual_ambig": False},
        {"k": 1, "label": "T", "manual_ambig": False},
        {"k": 2, "label": "S", "manual_ambig": False},
        {"k": 3, "label": "T", "manual_ambig": False},
    ]
    ps, pt, log, man = sequence_prior.apply_st_alternation_prior(ev, 48_000)
    assert ps == [0, 2] and pt == [1, 3]
    assert not man
    assert not [x for x in log if x.startswith("prior_drop")]


def test_prior_drops_spurious_t_before_s() -> None:
    # S, T, 多馀 T, S: 在期待下一段 S 时遇到多馀的 T 应被丢弃
    ev = [
        {"k": 0, "label": "S", "manual_ambig": False},
        {"k": 1, "label": "T", "manual_ambig": False},
        {"k": 2, "label": "T", "manual_ambig": False},
        {"k": 3, "label": "S", "manual_ambig": False},
    ]
    ps, pt, log, man = sequence_prior.apply_st_alternation_prior(ev, 48_000)
    assert ps == [0, 3] and pt == [1]
    assert any("prior_drop_t" in x for x in log)


def test_prior_early_s_manual() -> None:
    ev = [
        {"k": 0, "label": "S", "manual_ambig": False},
        {"k": 1, "label": "S", "manual_ambig": False},
    ]
    ps, pt, log, man = sequence_prior.apply_st_alternation_prior(ev, 48_000)
    assert ps == [0, 1] and not pt
    assert any(m.get("reason") == "s_without_prior_stop" for m in man)


def test_merged_events_require_label_key() -> None:
    ev = [{"k": 0, "label": "S", "manual_ambig": False}]
    ps, _, _, _ = sequence_prior.apply_st_alternation_prior(ev, 1000)
    assert ps == [0]
