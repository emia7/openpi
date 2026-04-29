"""
默认标定音与检测参数。播标与检标须共用同一份 `assets` WAV 或本模块参数经 `build_reference_*` 生成结果。

标准流程与参数含义以同目录 FREQ_WORKFLOW.md 为准。
"""

from __future__ import annotations

# 与常见视频/录音 48k 一致，减少重采样误差
SAMPLE_RATE_HZ: int = 48_000
# 标音时长 (s)，短音利于定位；窗长约 0.1–0.2s
BURST_DURATION_SEC: float = 0.12
# 上调频 chirp（开始）
START_CHIRP_F0_HZ: float = 2_200.0
START_CHIRP_F1_HZ: float = 4_400.0
# 下调频 chirp（结束）——与 start 在频域可分离
STOP_CHIRP_F0_HZ: float = 4_500.0
STOP_CHIRP_F1_HZ: float = 2_200.0

# 检峰
MIN_PEAK_DISTANCE_SEC: float = 0.35
# 与模板相关后的相对/绝对门限（v1 用「全局最大 × 比例」+ 最矮峰）
PEAK_SNR_RATIO: float = 0.35
# 对 ``max_corr * PEAK_SNR_RATIO`` 设上界，避免首段过强、后段真峰被滤（None=不限制）
PEAK_SNR_CAP: float | None = 0.24
ABS_CORR_FLOOR: float = 0.01
# 已废弃：现用 CROSS_CROSSTALK_MERGE_SEC + 两路找峰后合并
CROSS_TEMPLATE_DEDUPE_SEC: float = 0.0
# 两路 NCC 在「同一次」发声上各出一个峰、时间只错几 ms 时，用该秒数作链式合并，再比 rs/rt 定类
CROSS_CROSSTALK_MERGE_SEC: float = 0.07
# 簇内 max(rs@S) < max(rt@T) 时本判为「停」；若二者差值小于该阈值可改判为「开始」。
# 会改变时间轴上 S/T 交替，可能误把真停判成开、致成对异常；默认关闭。实机可极谨慎试 0.03~0.05。
PREFER_S_WHEN_T_WINS_BUT_MARGIN_BELOW: float | None = None
# 当 stop 比 start 多、出现 unpaired stop 时，在相邻区间用局部门限再扫 start 并插入
REFINE_UNPAIRED_STARTS: bool = True
# 再扫时：上一停之后、距本停之前，留足空白，避免和 stop 模板串扰
REFINE_AFTER_PREV_STOP_SEC: float = 0.04
REFINE_BEFORE_STOP_SEC: float = 0.14
# 局部门窗内 start NCC 须达到；过低时仍可能出片但质量差
WEAK_START_ABS_MIN: float = 0.10
WEAK_START_LO_CONFIDENCE: float = 0.20

# 成对
PAIR_TOLERANCE_SEC: float = 0.5
# 采集中脚踏/宏可能一次打出两个「开始」或两个「结束」；同键两次播标的最小间隔（秒），0=关闭
FOOT_KEY_DEBOUNCE_SEC: float = 0.35
# 切片在「停」标相关峰之后延长，把整段结束 chirp 录进 MP4；None 表示用与 stop 模板等长
INCLUDE_STOP_BEEP_TAIL_SEC: float | None = None
