"""record_marker_segments 的离线单测（不调用网络）。"""

from __future__ import annotations

import sys
from pathlib import Path

_NS = Path(__file__).resolve().parent
_UMI = _NS.parent
for d in (_NS, _UMI):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

from record_marker_segments import (  # noqa: E402
    marker_phrase_segments_from_char_timeline,
    matches_keywords,
    pair_record_clips,
    record_clips_to_jsonable,
)


def test_keywords() -> None:
    assert matches_keywords("好，开始 录制 了", ("开始", "录制"))
    assert not matches_keywords("只开始", ("开始", "录制"))


def test_one_clip_pair() -> None:
    w = {
        "text": "开始录制 停",
        "duration": 30.0,
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "开始录制"},
            {"start": 1.0, "end": 2.0, "text": "内容说了话"},
            {"start": 10.0, "end": 11.0, "text": "停止录制"},
        ],
    }
    clips, pw = pair_record_clips(
        w,
        30.0,
        after_start_sec=0.0,
        before_stop_sec=0.0,
        include_stop_mouth_sec=0.0,
    )
    assert len(clips) == 1
    c = clips[0]
    assert c.t_start_content == 0.0
    assert c.t_end_content == 10.0
    assert c.t_marker_stop_utter == 10.0
    j = record_clips_to_jsonable(clips)
    assert j[0]["index"] == 0
    assert not any("orphan" in x for x in pw)


def test_unclosed_eof() -> None:
    w = {
        "text": "开始",
        "duration": 5.0,
        "segments": [
            {"start": 0.0, "end": 0.5, "text": "开始录制"},
        ],
    }
    clips, pw = pair_record_clips(
        w,
        5.0,
        orphan_end="eof",
        after_start_sec=0.0,
        before_stop_sec=0.0,
        include_stop_mouth_sec=0.0,
    )
    assert len(clips) == 1
    assert clips[0].t_start_content == 0.0
    assert clips[0].t_end_content == 5.0
    assert any("unclosed" in x for x in pw)


def test_char_timeline_marker_segments() -> None:
    s = "开始录制。停止录制。开始录制。停止录制。开始录制。停止录制。"
    rows: list[dict[str, str | float]] = []
    t = 0.0
    for ch in s:
        rows.append({"t0": t, "t1": t + 0.1, "s": ch})
        t += 0.1
    w: dict = {
        "_filetrans_char_timeline": rows,
        "segments": [],
    }
    segs = marker_phrase_segments_from_char_timeline(
        w, start_keywords=("开始", "录制"), stop_keywords=("停止", "录制")
    )
    assert segs is not None
    assert len(segs) == 6
    clips, _pw = pair_record_clips(
        {**w, "segments": segs, "text": s, "duration": t},
        t,
        after_start_sec=0.0,
        before_stop_sec=0.0,
        include_stop_mouth_sec=0.0,
    )
    assert len(clips) == 3
    assert clips[0].t_start_content < clips[0].t_end_content

def test_include_stop_mouth_sec() -> None:
    w = {
        "text": "x",
        "duration": 30.0,
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "开始录制"},
            {"start": 9.0, "end": 9.2, "text": "停止录制"},
        ],
    }
    c0, _ = pair_record_clips(
        w,
        30.0,
        include_stop_mouth_sec=0.5,
        after_start_sec=0.0,
        before_stop_sec=0.0,
    )
    assert c0[0].t_end_content == 9.2
    assert any("clamped" in w for w in c0[0].warnings)
    c1, _ = pair_record_clips(
        w, 30.0, include_stop_mouth_sec=0.1, after_start_sec=0.0, before_stop_sec=0.0
    )
    assert c1[0].t_end_content == 9.1
    assert not c1[0].warnings


def test_stretched_leading_kai_refine_t0s() -> None:
    from record_marker_segments import _spans_phrase_from_per_char

    per = [
        ("开", 17.884, 21.404),
        ("始", 21.404, 21.404),
        ("录", 21.644, 21.884),
        ("制", 21.884, 22.124),
    ]
    s_off = _spans_phrase_from_per_char(
        per, "开始", "录制", refine_t0s_long_lead=False
    )
    s_on = _spans_phrase_from_per_char(
        per, "开始", "录制", refine_t0s_long_lead=True
    )
    assert len(s_off) == 1 and len(s_on) == 1
    assert s_off[0][0] == 17.884
    assert abs(s_on[0][0] - 21.644) < 1e-5


if __name__ == "__main__":
    test_stretched_leading_kai_refine_t0s()
    test_include_stop_mouth_sec()
    test_char_timeline_marker_segments()
    test_keywords()
    test_one_clip_pair()
    test_unclosed_eof()
    print("ok: test_record_marker_segments")
