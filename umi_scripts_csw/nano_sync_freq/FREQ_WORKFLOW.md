# 双音标定（nano_sync_freq）固定流程

与 `nano_sync`（口语 TTS + 百炼 ASR）**二选一**：本方案采集中只播 **chirp 标音**（非口语），后期**纯本地**互相关检峰 + `ffmpeg` 切片，**不依赖网络与 API Key**。

## 何时用哪套

| 场景 | 目录 / 文档 |
|------|-------------|
| 要 **chirp + 本地切分** | `nano_sync_freq/`，流程见**本文** |
| 要 **口语「开始/停止」+ ASR 时间戳** | `nano_sync/` + `RECORD_MARKERS_WORKFLOW.md` |

## 划分逻辑（实现要点）

1. **检峰**：对整段音轨分别与 `start` / `stop` 模板做滑动归一化互相关（NCC），在相关曲线上找局部极大；门限为 `max(corr) × PEAK_SNR_RATIO`，且不低于 `ABS_CORR_FLOOR`。  
2. **门限上界 `PEAK_SNR_CAP`（默认约 0.24）**：若**首几段**标音在音轨上特别强，全局 `max(corr)` 会很大，导致门限**过高**、**后段**同一套标音变弱时峰被漏掉。对门限做**上界限制**可找回中间/尾部的「停」等。  
3. **成对**（`pair_start_stop_times`）：按时间排序后，**每个「开始」配第一个严格在其之后的「停」**；多出来的停记 `unpaired_stop`。  
4. **补弱「开始」**（`refine_starts_for_unpaired_stops`）：若出现 `unpaired stop`，在**该停**与**前一停**之间再扫 `start` 模板：在局部时间窗内对 NCC 取 **argmax** 并做**抛物线子样本**；若窗内最大 NCC 仍低于 `WEAK_START_LO_CONFIDENCE`（`config.py`），会标出**低置信度**（尾段电平/环境更差时常见，**重采**比继续调门限更可靠）。  
5. **片尾**：`apply_stop_beep_tail` 把每段在「停」起音后再延约一帧 `stop` 长，但不超过**下一段开始**与**片尾**（见前序说明）。

CLI：`--no-peak-snr-cap` 关（1）的上界；`--no-weak-start-refine` 关（4）的补扫。

## 标准流程（按顺序执行）

### 1. 一次性：生成参考 WAV

在 **`umi_scripts_csw`** 下（改 `config.py` 或 `tone_gen` 参数后需重跑）：

```bash
cd /path/to/umi_scripts_csw
python3 nano_sync_freq/build_assets.py
```

生成 `nano_sync_freq/assets/start.wav`、`stop.wav`，与检测阶段**必须一致**。

### 2. 采集中：播标

**键位**（与 TTS 版一致）：**a = 开始一段**、**c = 结束一段**（播的是 chirp，不是口语）。

```bash
python3 nano_sync_freq/pedal_freq_listener.py
```

- 运行脚本的**终端需获焦**；脚踏若模拟键盘，按键需送到该终端。  
- 自定义：``--start-key x --stop-key z``。  
- 试音：``python3 nano_sync_freq/play_markers.py --self-test`` 或 ``-i``。

### 3. 后处理：检测 + 成对 + 切片

```bash
python3 nano_sync_freq/segment_by_freq_markers.py /path/to/record.mp4 \
  --cut-dir ./out_freq_clips \
  --markers-out ./out_freq_clips/markers.json
```

- **`--markers-out`**：建议每次固定写出，便于复现与排查。  
- 已有检测 JSON、只重跑配对/切片（改尾长等）：

```bash
python3 nano_sync_freq/segment_by_freq_markers.py /path/to/record.mp4 \
  --markers-in ./out_freq_clips/markers.json \
  --cut-dir ./out_freq_clips_recut
```

（注意：若 `--markers-in` 是**完整**一次跑出的文件，其中可能已含 `raw_detection`；若只是早期纯检测输出，需仍能对上同一条音轨。）

### 4. 片尾（结束标）行为（已固定为默认）

- **默认**：每段 `t_end` 在「**停**标起音」之后**再延长**一段，使 **整段结束 chirp 落在 clip 内**。  
- 延长量：优先 CLI ``--stop-tail-sec``，否则 `config.INCLUDE_STOP_BEEP_TAIL_SEC`（`None` 表示与 **`stop.wav` 模板等长**）。  
- **不要尾音、只切到停起音**：加 ``--no-stop-tail``。

JSON 中会写 `stop_beep_tail_sec`；每条 clip 有 `t_stop_onset`（停起音）与 `t_end`（含尾音后的结束时刻）。

### 5. 验收

```bash
python3 -m pytest nano_sync_freq/tests/ -q
```

实机：听 `out_freq_clips/clip_*.mp4` 与 `markers.json` 中时间是否一致；漏检/误检时调 `config.py` 中门限与 `MIN_PEAK_DISTANCE_SEC`（见 `README.md`「设计参数」）。

## 输出物约定

| 产物 | 说明 |
|------|------|
| `clip_0000.mp4`, … | 与源视频同容器扩展名时保留扩展名 |
| `markers.json`（若指定） | 含 `start_times`、`stop_times`、`pairing.clips[]`、`stop_beep_tail_sec`、`raw_detection`（全量跑时） |

## 设计参数与代码入口

- 全局默认：`config.py`（采样率、chirp 频带、检峰、成对容差、尾音 `INCLUDE_STOP_BEEP_TAIL_SEC`）。  
- 切片与 CLI：`segment_by_freq_markers.py`。

更细的依赖与模块说明见同目录 `README.md`。
