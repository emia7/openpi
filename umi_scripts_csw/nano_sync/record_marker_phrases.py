"""
采集端 TTS 与百炼 filetrans ASR 后处理共用的「录制起止」短语文案与关键词。

修改此处即可同时影响同目录下 record_marker_tts_listener 与 segment_by_record_markers。
"""

from __future__ import annotations

# 与主机 TTS 念出的短句尽量一致，便于 ASR 命中
DEFAULT_START_PHRASE = "开始录制"
DEFAULT_STOP_PHRASE = "停止录制"

# 后处理匹配：在去掉空白后，每段需**同时**包含这些子串（可放宽误识别）
DEFAULT_START_KEYWORDS: tuple[str, str] = ("开始", "录制")
DEFAULT_STOP_KEYWORDS: tuple[str, str] = ("停止", "录制")
