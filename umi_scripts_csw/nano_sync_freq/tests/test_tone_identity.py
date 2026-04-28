"""A. 标音一致性。"""

from __future__ import annotations

import numpy as np

from nano_sync_freq import config
from nano_sync_freq.tone_gen import (
    build_reference_samples,
    stable_reference_digest,
)


def test_stable_reference_digest_unchanged_across_calls() -> None:
    a = stable_reference_digest()
    b = stable_reference_digest()
    assert a == b, "同参生成应稳定，否则播/检会漂移"
    assert len(a) == 32


def test_start_stop_differ() -> None:
    a = build_reference_samples("start")
    b = build_reference_samples("stop")
    assert a.shape == b.shape
    assert float(np.sqrt(np.mean((a - b) ** 2))) > 0.01


def test_repeated_build_identical() -> None:
    x0 = build_reference_samples("start")
    x1 = build_reference_samples("start")
    assert np.array_equal(x0, x1)
    assert int(config.SAMPLE_RATE_HZ) == 48_000
