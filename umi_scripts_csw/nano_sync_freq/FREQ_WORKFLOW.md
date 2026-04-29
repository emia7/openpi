# 双音标定（nano_sync_freq）固定流程

与 `nano_sync`（口语 TTS + 百炼 ASR）**二选一**：本方案采集中只播 **chirp 标音**（非口语），后期**纯本地**互相关检峰 + `ffmpeg` 切片，**不依赖网络与 API Key**。

**本文档为团队默认流程的单一事实来源（SSoT）**；`README.md` 只作依赖与参数索引。

---

## 何时用哪套

| 场景 | 目录 / 文档 |
|------|-------------|
| 要 **chirp + 本地切分** | `nano_sync_freq/`，**以本文为准** |
| 要 **口语「开始/停止」+ ASR 时间戳** | `nano_sync/` + `RECORD_MARKERS_WORKFLOW.md` |

---

## 采集约定（必须一致）

1. **键位**（与播标脚本一致）：**a = 开始一段**、**c = 结束一段**；播的是**两种不同 chirp**（`start.wav` / `stop.wav`），不是口语。  
2. **时间轴上第一条可检标**（整段音轨的**第一簇**开始/停串扰）按团队约定视为 **「开始」**；检测里用 `FAVOR_S_FOR_FIRST_MERGED_CLUSTER` 落在此约定。若某次录像是**故意以「停」开镜**，应关闭该选项或改后处理。  
3. 片段语义：**每段 clip = 从某次「开始」到其后第一次「停」**（成对规则见下），与采集中「先 a 开段、再 c 停段」一致。

---

## 划分逻辑（实现固定版）

### 1. 互相关检峰

对整段音轨分别与 `start` / `stop` 模板做滑动归一化 NCC，得到 `rs[i]`、`rt[i]`。在**两路各自**上找局部极大（`MIN_PEAK_DISTANCE_SEC` 防过密峰）。

门限：约 `min(max(corr)×PEAK_SNR_RATIO, PEAK_SNR_CAP)` 与 `ABS_CORR_FLOOR` 的较大者。  
`PEAK_SNR_CAP`：避免首段标音过强抬高压掉后段弱峰，**不要**随意关掉（除非用 `--no-peak-snr-cap` 做对照）。

### 2. 串扰合并 + 定类（核心）

**同一次按键**在 `rs`、`rt` 上各常出一个峰，时间仅差数 ms～数十 ms，不可当两次事件。

- 将两路峰**按下标**排序，把「相邻下标时间差 ≤ `CROSS_CROSSTALK_MERGE_SEC`」的峰**链式**合成一簇。  
- **定类**（**禁止**在簇里用「同一下标上 `rs[k] vs rt[k]`」主判，会串台）：在簇内  
  - 在**属于 start 路峰**的下标上取 `max(rs[i])`；  
  - 在**属于 stop 路峰**的下标上取 `max(rt[j])`；  
  - 两数比较，大者定类；在**获胜那一路**里取 NCC 最大的下标为时间戳。  
- **整段第一簇**且 S/T 两路都有峰时，按采集约定**强制记为开始**（`FAVOR_S_FOR_FIRST_MERGED_CLUSTER`），避免首包停模板略大就判成「停」。

> `CROSS_CROSSTALK_MERGE_SEC` 过小会拆成两次事件；过小典型 ~0.02 s 可能合并不了 ~27 ms 的双峰。默认 **0.07 s** 在常见素材上更稳，可按现场再调。

### 3. 成对

`pair_start_stop_times`：时间排序后，**每个「开始」配第一个严格在其之后的「停」**；多停 → `unpaired_stop` 等 warning。

### 4. 补弱「开始」

`refine_starts_for_unpaired_stops`：仅当出现「多一个停、少一个起」时，在**相邻停之间**的局部窗对 **start 模板**再 argmax。可用 `--no-weak-start-refine` 关。

### 5. 片尾

`apply_stop_beep_tail`：在「停」**起音**后**再**延长一段，使**整段结束 chirp**落在 clip 内，且**不超过**下一段开始 / 片尾。延长量见下「CLI」。

### 6. 切片

`segment_by_freq_markers.py` 用 `ffmpeg` **流复制**（`-c copy`）从原视频切 `[t_start, t_end]`。时间来自成对 + 尾长。

相关代码入口：`detect_tones.py`（合并在 `_merge_opposing_peaks_to_one_label`）、`pair_markers.py`、`segment_by_freq_markers.py`、`config.py`。

---

## 标准操作顺序（照做即可）

### 一次性：生成参考 WAV

在 **`umi_scripts_csw`** 下（**改** `config.py` 里标音参数后必须重跑）：

```bash
cd /path/to/umi_scripts_csw
python3 nano_sync_freq/build_assets.py
```

得到 `nano_sync_freq/assets/start.wav`、`stop.wav`，**播标与后处理检测必须同一份**。

### 采集中：播标

```bash
python3 nano_sync_freq/pedal_freq_listener.py
# 需不抢终端焦点时（要 pynput；Linux Wayland 常不行）:
python3 nano_sync_freq/pedal_freq_listener.py --global
```

- 让运行脚本的**终端获焦**后再按 a/c。  
- 试音：`python3 nano_sync_freq/play_markers.py --self-test` 或 `-i`。

### 后处理：检测 + 成对 + 切片 + 落盘

**团队默认**（**务必**带 `markers.json` 便于复现/排查）：

```bash
python3 nano_sync_freq/segment_by_freq_markers.py /path/to/record.mp4 \
  --cut-dir ./out_freq_clips \
  --markers-out ./out_freq_clips/markers.json
```

- **只要 JSON、不切片**：`--write-markers-only` 且指定 `--markers-out`；或省略 `--cut-dir` 看标准输出。  
- **只切到「停」起音、不要尾长**：`--no-stop-tail`。  
- **覆盖尾长**（秒）：`--stop-tail-sec 0.12` 或设 `config.INCLUDE_STOP_BEEP_TAIL_SEC`。  
- **重跑切分、不改检测**（如只改尾长 / 用存好的 `markers-in` 里同轨时间）：

```bash
python3 nano_sync_freq/segment_by_freq_markers.py /path/to/record.mp4 \
  --markers-in ./out_freq_clips/markers.json \
  --cut-dir ./out_freq_clips_recut \
  --no-stop-tail
```

常用对照：`--no-peak-snr-cap`、`--no-weak-start-refine` 见上。

### 验收

```bash
cd /path/to/umi_scripts_csw
python3 -m pytest nano_sync_freq/tests/ -q
```

实机：听 `clip_000*.mp4`，对照 `markers.json` 中 `start_times` / `stop_times` 与 `pairing.clips`。

### 段数与手数对不上时（如「心里数 129 次、只切出 109 段」）

- **不是**单路 NCC 普遍漏峰：在一条远场/机身麦长片上，**开始路**上常见仍有 **200+ 个**原始峰，合并后每簇要判成「起」或「停」；若**起**的 chirp 比**停**略弱，会出现大量「双路都有的簇里停险胜」→ 成对后 **「起」比「停」少一截**（你看到的 109 对正是这么来的）。**误触**、多检的停峰会表现为 `drop_stop_before_or_at_start` 等；与「少段」是两类现象。  
- `config` 里可选 `PREFER_S_WHEN_T_WINS_BUT_MARGIN_BELOW`：在「停险胜但差值极小」时改判为**起**。**不能**当万能默认：可能把真停判成起、时间轴上**连 S**，成对后反而 `unpaired_start` 或段数更乱；仅作调参实验。更稳的是**改麦位/电平、重录参考、或**做 a/c 交替的时序模型（未在 v1 里实现）。

---

## 输出物

| 产物 | 说明 |
|------|------|
| `clip_0000.mp4`, … | 与源同扩展名时沿用 |
| `markers.json`（若 `--markers-out`） | 含 `start_times`、`stop_times`、`pairing`、`stop_beep_tail_sec`、警告；**全量从检测跑时**可含 `raw_detection` |

---

## 与实现绑定的 `config` 项（需改时只动这里 + 重跑 `build_assets` 若动标音）

| 项 | 作用 |
|----|------|
| `CROSS_CROSSTALK_MERGE_SEC` | 两路 NCC 上「同一声」两峰的合并时间窗（秒） |
| `FAVOR_S_FOR_FIRST_MERGED_CLUSTER` | 第一簇 S+T 并存时是否**强制**记为开始 |
| `PEAK_SNR_CAP` / `PEAK_SNR_RATIO` / `MIN_PEAK_DISTANCE_SEC` / `ABS_CORR_FLOOR` | 门限与峰距 |
| `PREFER_S_WHEN_T_WINS_BUT_MARGIN_BELOW` | 险胜时改判为起（**默认关**；见上「段数与手数」） |
| `REFINE_UNPAIRED_STARTS` 及 `WEAK_START_*` | 补弱开始 |
| `INCLUDE_STOP_BEEP_TAIL_SEC` | 停起音后延长；`None` = 与 `stop` 模板等长 |

---

## 与 `nano_sync` 的边界

| 需求 | 使用 |
|------|------|
| 脚踏 A/C 播**标定 chirp** + 本流程切分 | `nano_sync_freq` + **本文** |
| 脚踏 A/C 播**口语** + 百炼 ASR | `nano_sync` + `RECORD_MARKERS_WORKFLOW.md` |

更细的播标依赖、试音、模块列表见同目录 `README.md`。
