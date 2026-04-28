# 双音标定分段（nano_sync_freq）

与 `nano_sync` 的 **ASR/口语锚点** 方案**完全独立**：采集中播放**两种**不同线性 chirp（`assets/start.wav`、`assets/stop.wav`），后期在音轨上做**归一化互相关**检峰，再 `ffmpeg` 切片。

**固定流程（步骤、片尾约定、与 `nano_sync` 对照）以 [`FREQ_WORKFLOW.md`](FREQ_WORKFLOW.md) 为准**；本文件补充依赖、试音与参数细节。

## 依赖

- Python 3.9+、`numpy`
- `ffmpeg` 或 `pip install imageio-ffmpeg`（与 `audio_extract` 一致）
- 播标：mac 推荐系统自带 `afplay`；否则 `ffplay` 或 `pip install sounddevice soundfile`
- **采集按键**：`pedal_freq_listener.py` **只监听当前终端**，无需 `pynput`（与 TTS 全局监听版不同）

## 一次性：生成参考音

在 **`umi_scripts_csw`** 下：

```bash
python3 nano_sync_freq/build_assets.py
# 或覆盖：python3 nano_sync_freq/build_assets.py --force
```

## 采集中

与 `record_marker_tts_listener` **同一套键位**：**a = 开始一段**、**c = 结束一段**（播 chirp，不播口语）。

**本终端监听（默认，无 pynput）**：

```bash
cd /path/to/umi_scripts_csw
python3 nano_sync_freq/build_assets.py    # 首次或改参后
python3 nano_sync_freq/pedal_freq_listener.py
```

- 先**让该终端窗口获得焦点**，再按 **a** / **c**（macOS / Linux 一般单键即响；Windows 为每行一个字母后回车）。  
- 脚踏若模拟键盘，需把按键送到**正在运行脚本的终端**（焦点在该窗口时踏 A / 踏 C）。  
- 自定义键位：``--start-key x --stop-key z``（单字符）。

**仅试扬声器**（不监听键）：

```bash
python3 nano_sync_freq/play_markers.py --self-test
```

终端里用 `s`/`t` 试音（`play_markers.py -i`，与 a/c 键位无关，仅自测）：

```bash
python3 nano_sync_freq/play_markers.py -i
```

也可单次：``--start`` / ``--stop``。

## 后处理：检测 + 切片

**推荐**（同时写出 `markers.json` 便于复现）：

```bash
python3 nano_sync_freq/segment_by_freq_markers.py /path/to/record.mp4 \
  --cut-dir ./out_clips --markers-out ./out_clips/markers.json
```

- **片尾**：默认在「停」标起音后再延长，把**整段结束 chirp** 录进每段；延长量与 `stop` 模板等长，除非在 `config.py` 设 `INCLUDE_STOP_BEEP_TAIL_SEC` 或用 CLI ``--stop-tail-sec``。只要切到起音、不要尾音：``--no-stop-tail``。  
- 只输出 JSON、不切片：省略 ``--cut-dir``，或加 ``--write-markers-only``。  
- 使用已保存的检测 JSON（少重算互相关）：

```bash
python3 nano_sync_freq/segment_by_freq_markers.py /path/to/record.mp4 \
  --markers-in ./markers.json --cut-dir ./out_clips
```

切片输出目录建议放 `nano_sync_freq/out_*/`（已加入根 `.gitignore`）或仓库外路径。

## 与 `nano_sync` 的边界

| 需求 | 使用 |
|------|------|
| 脚踏 A/C 播**标定 chirp**（进后期互相关切分） | `nano_sync_freq/pedal_freq_listener.py` |
| 脚踏 A/C 播**口语 TTS**「开始/停止录制」 | `nano_sync/record_marker_tts_listener.py` |
| 百炼 ASR 按口语切分 | `nano_sync` + `RECORD_MARKERS_WORKFLOW.md` |

## 测试

在 `umi_scripts_csw` 下：

```bash
python3 -m pytest nano_sync_freq/tests/ -q
```

阶段说明见计划中的「测试计划」：合参标音一致、合成检测、鲁棒、成对逻辑、WAV roundtrip；**实机烟测**（实麦 + 本机播标）在交付前自行做。

## 设计参数

见 `config.py`：48 kHz、标音约 0.12 s、开始为上调频 chirp、结束为下调频 chirp，检峰门限与最小峰距可随环境再调。成对容差 `PAIR_TOLERANCE_SEC`；**片尾**默认 `INCLUDE_STOP_BEEP_TAIL_SEC = None`（用与 `stop` 模板等长的延长）。
