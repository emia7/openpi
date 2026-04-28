"""qwen_filetrans_asr：离线校验 transcripts→segments 映射。"""

from __future__ import annotations

import sys
from pathlib import Path

_NS = Path(__file__).resolve().parent
_UMI = _NS.parent
for d in (_NS, _UMI):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

from qwen_filetrans_asr import transcription_doc_to_whisper_shape  # noqa: E402


def test_transcription_to_segments_ms_to_sec() -> None:
    doc = {
        "file_url": "https://example.com/a.wav",
        "transcripts": [
            {
                "channel_id": 0,
                "text": "开始录制。内容。停止录制。",
                "sentences": [
                    {"sentence_id": 0, "begin_time": 0, "end_time": 1000, "text": "开始录制。"},
                    {"sentence_id": 1, "begin_time": 1000, "end_time": 5000, "text": "内容。"},
                    {"sentence_id": 2, "begin_time": 10000, "end_time": 11000, "text": "停止录制。"},
                ],
            }
        ],
    }
    w = transcription_doc_to_whisper_shape(doc)
    assert w["duration"] == 11.0
    assert len(w["segments"]) == 3
    assert w["segments"][0]["start"] == 0.0 and w["segments"][0]["end"] == 1.0
    assert w["segments"][2]["start"] == 10.0
    assert "开始录制" in w["text"]


if __name__ == "__main__":
    test_transcription_to_segments_ms_to_sec()
    print("ok: test_qwen_filetrans_normalize")
