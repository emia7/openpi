# UMI Scripts 目录说明

> 本目录包含数据处理、转换、评估的各类工具脚本，按功能分类组织。

## 目录结构

```
umi_scripts_csw/
├── README.md                          # 本文件
├── INDEX.md                           # 脚本功能索引
├── data_detection/                    # 数据检测 (ArUco/Foundation Pose)
├── data_cleaning/                     # 数据清理与质量检查
├── data_conversion/                   # 数据格式转换
├── data_evaluation/                   # 数据评估与可视化
├── rosbag_tools/                      # ROSBag处理工具
├── nano_sync/                         # 脚踏 TTS 口语锚点 + 百炼 Qwen ASR 切分
├── nano_sync_freq/                    # 双 chirp 标音 + 本地互相关 + ffmpeg 切分（见 FREQ_WORKFLOW.md）
└── utils/                             # 通用工具脚本
```

## 各目录说明

### data_detection/ - 数据检测
**用途**: 第三视角视觉检测，提取机械臂位姿

| 脚本 | 功能 |
|------|------|
| add_aruco_pose_to_json.py | ArUco码检测，输出left_in_cam/right_in_cam到third.json |
| visualize_aruco_detection.py | 可视化ArUco检测结果，生成检测框和坐标轴预览图 |

**输入**: third.mp4  
**输出**: third.json (含检测位姿)

---

### data_cleaning/ - 数据清理
**用途**: 数据质量检查、异常筛选、合并清理

| 脚本 | 功能 | 推荐用法 |
|------|------|---------|
| **check_trajectory_anomalies.py** | **轨迹异常检测** (完整分析，支持CHECK-004+~010) | Step 1: 检测异常 |
| **generate_visual_report.py** | **可视化报告生成** (HTML/Markdown) | Step 2: 生成报告 |
| **execute_data_cleaning.py** | **执行数据清理** (删+重编号) | Step 3: 正式清理 |
| clean_data_0422_preview.py | 旧版清理预览 (仅供参考) | - |
| check_consistency.py | 检查数据集内部一致性 (视频分辨率、文件完整性) | - |
| check_dataset_actions.py | 检查数据集动作统计分布 | - |
| compare_batches.py | 对比两个数据批次的差异 | - |
| compare_npz.py | 对比NPZ数据文件 | - |

**快速开始 - 标准清理流程**:
```bash
# Step 1: 检测异常
python data_cleaning/check_trajectory_anomalies.py \
    --data_dir ~/Downloads/handover_umi_0422 \
    --output_dir ./anomaly_reports

# Step 2: 生成可视化报告
python data_cleaning/generate_visual_report.py \
    --json_report ./anomaly_reports/trajectory_anomaly_report.json \
    --format both

# Step 3: 预览清理
python data_cleaning/execute_data_cleaning.py \
    --data_dir ~/Downloads/handover_umi_0422 \
    --anomaly_report ./anomaly_reports/trajectory_anomaly_report.json \
    --severity critical \
    --dry_run

# Step 4: 正式清理
python data_cleaning/execute_data_cleaning.py \
    --data_dir ~/Downloads/handover_umi_0422 \
    --anomaly_report ./anomaly_reports/trajectory_anomaly_report.json \
    --severity critical \
    --output_dir ~/Downloads/handover_umi_0422_cleaned
```

---

### data_conversion/ - 数据转换
**用途**: 转换为LeRobot格式或其他格式

| 脚本 | 功能 |
|------|------|
| convert_mp4_data_to_lerobot_123.py | 原转换脚本 (支持相对旋转state) |
| convert_mp4_data_to_lerobot_123_aruco.py | **ArUco版本** (支持hands_rel_xyz + hands_rel_rot6d 9D state) |
| convert_mp4_data_to_lerobot_downsample.py | 下采样版本 |
| convert_mp4_data_to_lerobot_downsample_13.py | 下采样版本 (13fps) |

**输入**: MP4 + JSON (stage1格式)  
**输出**: LeRobot格式数据集

---

### data_evaluation/ - 数据评估
**用途**: 评估数据质量、可视化轨迹

| 脚本 | 功能 |
|------|------|
| eval_actions.py | 动作序列评估 |
| eval_relative.py | 相对位姿评估 |
| eval_dual_relative.py | 双手相对位姿评估 |
| eval_relative_visualize.py | 相对位姿可视化 |

---

### rosbag_tools/ - ROSBag工具
**用途**: ROSBag录制数据处理

| 脚本 | 功能 |
|------|------|
| convert_rosbag_to_mp4_vis_123.py | 主版本: rosbag转MP4+JSON (3视角) |
| convert_rosbag_to_mp4_vis.py | 早期版本 |
| convert_rosbag_to_mp4_vis_13.py | 13fps版本 |
| convert_ros_data_to_mp4.py | 简化版本 |

**输入**: .bag  
**输出**: MP4 + JSON (stage1格式)

---

### nano_sync_freq/ - 双 chirp 标定 + 本地切分

**用途**：采集中 **a/c** 播**两种 chirp**（非 TTS 口语），后期**纯本地**检峰 + 切片，**不依赖百炼与公网 URL**。

| 文档 / 入口 | 说明 |
|--------------|------|
| [nano_sync_freq/FREQ_WORKFLOW.md](nano_sync_freq/FREQ_WORKFLOW.md) | **固定流程**（建 assets → 采集中 `pedal_freq_listener` → `segment_by_freq_markers` + 片尾约定） |
| [nano_sync_freq/README.md](nano_sync_freq/README.md) | 依赖、试音、设计参数 |
| [nano_sync_freq/segment_by_freq_markers.py](nano_sync_freq/segment_by_freq_markers.py) | 检测 + 成对 + `ffmpeg` 切 `clip_*.mp4` |

**推荐后处理**（同目录下）：

```bash
cd umi_scripts_csw
python3 nano_sync_freq/segment_by_freq_markers.py path/to/record.mp4 \
  --cut-dir ./nano_sync_freq/out_freq_clips --markers-out ./nano_sync_freq/out_freq_clips/markers.json
```

与 **nano_sync**（TTS + ASR）的选用对照见 `FREQ_WORKFLOW.md` 文首表。

---

### nano_sync/ - 音视频同步 / 录制 TTS 锚点 + Qwen 切分
**用途**: 脚踏板与主机口令同步；ASR 使用阿里云百炼 **DashScope** `qwen3-asr-flash-filetrans`（异步、句级时间戳）。**蜂鸣旧方案已删除**（仅保留 TTS + 百炼 ASR）。

| 脚本/模块 | 作用 |
|----------|------|
| [record_marker_phrases.py](nano_sync/record_marker_phrases.py) | 与采集 TTS/后处理共用的「开始/停止录制」文案与关键词 |
| [qwen_filetrans_asr.py](nano_sync/qwen_filetrans_asr.py) | 百炼异步 filetrans：提交任务、轮询、`transcription_url` 下载并映射为 `segments` |
| [upload_wav_oss.py](nano_sync/upload_wav_oss.py) | 将导出的音轨上传至**阿里云 OSS**，输出公网 `https://` URL 供 ``--file-url``（需 `pip install oss2`） |
| [record_marker_tts_listener.py](nano_sync/record_marker_tts_listener.py) | 采集中 a/c 触发 TTS（macOS `say` / Win PowerShell SAPI / Linux espeak）；`--self-test`；**无 pynput 时** `--use-stdin`；否则 `python3 -m pip install pynput` |
| [segment_by_record_markers.py](nano_sync/segment_by_record_markers.py) | 本地 `path` + 公网 ``--file-url``（与音轨对齐）→ `clips` JSON；可选 ``--cut-dir``+ffmpeg；`--from-json` 离线调试 |
| [record_marker_segments.py](nano_sync/record_marker_segments.py) | 从 `segments` 做锚点配对 |
| [test_record_marker_segments.py](nano_sync/test_record_marker_segments.py) | 离线单测，无网络 |
| [test_qwen_filetrans_normalize.py](nano_sync/test_qwen_filetrans_normalize.py) | 离线校验 filetrans JSON → `segments` 映射 |

**环境**：在 `umi_scripts_csw/.env` 写入 `DASHSCOPE_API_KEY=sk-...`（勿提交）。国际地域可设 `DASHSCOPE_API_BASE=https://dashscope-intl.aliyuncs.com/api/v1`。

**`--file-url`**：filetrans 要求音频为**公网 HTTPS 可直链访问**（与本地 `path` 音轨一致）。典型流程：从长视频**抽出 WAV**（见下方 E2E）→ **上传 OSS** 等对象存储 → 将脚本打印的 URL 传入 ``--file-url``。仅调试分段逻辑时用 `--from-json` 跳过 API。`upload_wav_oss.py` 执行时会**自动加载** `umi_scripts_csw/.env`（与百炼 Key 同文件；OSS 的 AK/SK 也写这里即可）。

**与腕部（FastUMI 等）的配对**：本仓库**不提供**「nano 切段与腕部视角」的自动时间对齐脚本；请按各设备时间轴在标注或后处理中**人工对齐**。

#### 采集端三系统（T5）

| 系统 | TTS | 键盘 |
|------|-----|------|
| macOS | `say`（中文优先 Tingting 等） | `python3 -m pip install pynput`；装不上或权限问题时用 `--use-stdin` |
| Windows | PowerShell `System.Speech`（`--self-test` 即可验） | 同上；`--use-stdin` 为每行输入 a/c 后回车 |
| Linux | `espeak-ng` / `espeak` / `spd-say`（如 `sudo apt install espeak-ng`） | 同上；`--use-stdin` 在 TTY 下单键 |

长视频多轮任务验收：任一下载到本机的 mp4，抽同轨 WAV → 上传得 `https` → `segment_by_record_markers --file-url`；**clips 条数应等于录制轮次**（如三轮开始/停止→3 条）；可 `--asr-only` 排查 ASR。

#### nano 无本体：采集 → 公网音轨 → 分段（E2E）

在仓库根下将 `umi_scripts_csw` 为当前工作目录，以下命令可逐行复制（路径请替换为真实文件）：

```bash
cd umi_scripts_csw

# 1) 采集中：脚踏 → 主机 TTS（「开始录制」/「停止录制」）→ nano 一条长录（见 record_marker_tts_listener --self-test）

# 2) 从长视频导出与画面对齐的 WAV（filetrans 需要与本地 path 为同一条音轨；亦可直接用 ffprobe/ ffmpeg）
python3 -c "
from pathlib import Path
from audio_extract import probe_audio_with_ffmpeg, extract_audio_wav
p = Path('path/to/nano_session.mp4')
o = Path('/tmp/nano_session.wav')
probe_audio_with_ffmpeg(p)
extract_audio_wav(p, o, sample_rate=16000, mono=True)
print('wrote', o)
"

# 3) 上传得稳定 HTTPS URL（配置 OSS 环境变量后；依赖 pip install oss2）
#    python3 nano_sync/upload_wav_oss.py /tmp/nano_session.wav

# 4) 百炼 filetrans + 按口令出 clips
#    python3 nano_sync/segment_by_record_markers.py path/to/nano_session.mp4 \\
#        --file-url 'https://your-bucket.oss-xxx.aliyuncs.com/.../nano_session.wav' \\
#        --out-json /tmp/nano_clips.json

# 5) 可选：按时间窗裁片
#    python3 nano_sync/segment_by_record_markers.py path/to/nano_session.mp4 \\
#        --file-url '...同上...' --out-json /tmp/nano_clips.json --cut-dir ./cuts
```

快速抽查画面/音轨统计（**不**走百炼）：

```bash
python3 play_video_check_audio.py path/to/nano_session.mp4 --extract-only
```

**分段门禁 T6（`--from-json`）**：首次需生成本地静音占位 WAV（**不提交 git**）：

```bash
python3 nano_sync/fixtures/ensure_t6_wav.py
python3 nano_sync/segment_by_record_markers.py nano_sync/fixtures/t6_stub.wav \\
  --from-json nano_sync/fixtures/t6_stub_asr.json --out-json /tmp/clips.json
```

| `upload_wav_oss` 新增/变更 | 抽轨自测（T4）+ 带 Key 的 filetrans 全链路（T7、建议 T8） |

---

### utils/ - 通用工具
**用途**: 各类辅助工具

| 脚本 | 功能 |
|------|------|
| json_sort.py / json_sort_123.py / json_sort_13.py | JSON文件排序整理 |
| json_visualize.py | JSON数据可视化 |
| transform_pose.py | 位姿转换工具 |
| camera_test_analyzer.py | 相机测试分析 |
| tri_image_sampler_10hz.py | 三视角图像采样 |
| render_triad_mp4.py | 渲染triad视频 |
| replay_data_fastumi.py | FastUMI数据回放 |
| fastumi_helper.py | FastUMI辅助工具 |
| test.py | 测试脚本 |

---

## 数据处理流程

### 完整流程 (含清理)

```
原始数据 (MP4+JSON)
    ↓ [data_cleaning/check_trajectory_anomalies.py]
异常检测报告 (JSON)
    ↓ [data_cleaning/generate_visual_report.py]
可视化报告 (HTML)
    ↓ (查看报告，确认清理级别)
执行清理 [data_cleaning/execute_data_cleaning.py]
    ↓
清理后数据 (cleaned/)
    ↓ [data_conversion/]
LeRobot格式数据集
```

### 旧流程 (ArUco检测，非必须)

```
rosbag (.bag)
    ↓ [rosbag_tools/]
MP4 + JSON (stage1)
    ↓ [data_detection/] (可选)
MP4 + JSON (含检测数据)
    ↓ [data_conversion/]
LeRobot格式数据集
    ↓ [data_evaluation/]
质量评估报告
```

## 快速开始

### 标准数据清理流程 (推荐，从原始数据到清理后数据)

```bash
# Step 1: 轨迹异常检测 (CHECK-004+ ~ CHECK-010)
python data_cleaning/check_trajectory_anomalies.py \
    --data_dir ~/Downloads/handover_umi_0422 \
    --output_dir ./anomaly_reports

# Step 2: 生成可视化报告 (查看HTML决定清理级别)
python data_cleaning/generate_visual_report.py \
    --json_report ./anomaly_reports/trajectory_anomaly_report.json \
    --output_dir ./anomaly_reports \
    --format both
# 查看: anomaly_reports/trajectory_anomaly_visual.html

# Step 3: 预览清理 (确认要删除的episodes)
python data_cleaning/execute_data_cleaning.py \
    --data_dir ~/Downloads/handover_umi_0422 \
    --anomaly_report ./anomaly_reports/trajectory_anomaly_report.json \
    --severity critical \
    --dry_run

# Step 4: 正式执行清理 (删除异常并重新编号)
python data_cleaning/execute_data_cleaning.py \
    --data_dir ~/Downloads/handover_umi_0422 \
    --anomaly_report ./anomaly_reports/trajectory_anomaly_report.json \
    --severity critical \
    --output_dir ~/Downloads/handover_umi_0422_cleaned

# Step 5: 验证结果
ls ~/Downloads/handover_umi_0422_cleaned/ | grep "_left.json$" | wc -l
# 应显示清理后的episodes数量
cat ~/Downloads/handover_umi_0422_cleaned/cleaning_log.json
# 查看清理日志
```

### 完整数据处理流程 (含ArUco，可选)

```bash
# 1. rosbag转MP4 (如果是rosbag源数据)
python rosbag_tools/convert_rosbag_to_mp4_vis_123.py \
    --bag_dir /path/to/bags \
    --out_dir /path/to/stage1 \
    --start_idx 1

# 2. ArUco检测 (如果需要hands_rel数据)
python data_detection/add_aruco_pose_to_json.py \
    --data_dir /path/to/stage1 \
    --output_dir /path/to/stage1_aruco

# 3-7. 清理流程 (同上)
# ...

# 8. 转换为LeRobot
python data_conversion/convert_mp4_data_to_lerobot_123_aruco.py \
    --stage1_dir /path/to/stage1_aruco \
    --repo my_dataset
```

### 2. 数据质量检查

```bash
# 检查一致性
python data_cleaning/check_consistency.py --data_dir /path/to/data

# 评估数据
python data_evaluation/eval_dual_relative.py --data_dir /path/to/data
```

## 规范文档

- [数据合并与清理规范](../docs/data_merge_clean_spec.md) - 定义数据清理规则
- [清理报告示例](../docs/cleaning_report_0422_preview.json) - 0422数据清理预览报告

## 注意事项

1. **新脚本放置**: 按功能放入对应目录，不要在根目录直接创建
2. **命名规范**: 使用 `{功能}_{描述}_{版本/变体}.py` 格式
3. **版本管理**: 不同变体用 `_123`, `_13` 等后缀区分
4. **文档维护**: 修改本README和INDEX.md记录新脚本
