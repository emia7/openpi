# 录制口令分段流程（飞书等长视频 → 多段 MP4）

本文档**存档**当前（2026-04）在 `nano_sync` 中用于「按**开始录制 / 停止录制**切多段内容」的端到端流程、依赖与调参位置，便于复现与后续改动对照。

若采集中使用 **chirp 标音**（非口语）且**不需要 ASR**，请改用 **`nano_sync_freq/FREQ_WORKFLOW.md`**，与本流程二选一。

## 目标

- 对一条音画同步的媒体（如飞书导出的 `*.mp4`），依口播锚点切出**若干段** `clip_0000.mp4 …`。
- 每段**时间语义**（默认）：
  - 起点：该轮 **「开始录制」** 句子的**可剪起点**（见下「字级与启发式」）。
  - 终点：该轮 **「停止录制」** 句**起音**之后，可再带一小段口型（`--include-stop-mouth-sec`），且不超过该**停止句末**、不超过媒体末。

## 依赖与凭据

| 项 | 说明 |
|----|------|
| `ffmpeg` | 系统 `ffmpeg` 或 `imageio-ffmpeg` 提供的二进制；`audio_extract.find_ffmpeg()` 会解析。 |
| 百炼 DashScope | 环境变量 `DASHSCOPE_API_KEY`；可放在 `umi_scripts_csw/.env` 或 `nano_sync/.env`（**勿提交**）。 |
| 公网 `file_url` | `qwen3-asr-flash-filetrans` 只接受**可直链 GET 的 HTTPS 音频**；与本地 `path` 音轨需为同一段。 |
| 字级时间戳 | 转写时打开 **`--enable-words`**，结果中才会有 `_filetrans_char_timeline`，用于多轮口令对齐。 |

## 核心脚本与数据流

```
本地视频/音频
    → 抽音轨（audio_extract.extract_audio_wav 等）→ 上传 OSS 或临时公网直链
    → segment_by_record_markers.py
           ├ 调用 run_filetrans_to_whisper_dict(..., enable_words=True)
           ├ 用 _filetrans_char_timeline 重排为交替「开始/停止」句
           │     见 record_marker_segments.marker_phrase_segments_from_char_timeline
           ├ 配对片窗 record_marker_segments.pair_record_clips
           │     （含 include_stop_mouth_sec）
           └ 可选 --cut-dir：ffmpeg stream copy 裁 MP4
```

- **ASR 映射**：`qwen_filetrans_asr.py`（`transcription_doc_to_whisper_shape` 会附带 `_filetrans_char_timeline`）。
- **对口播合并句**：若有字级时间轴则**不用** `expand_merged_start_stop_utterances` 做比例切分，否则多轮会不可靠。
- **片段时间**：`record_marker_segments.py` 中 `pair_record_clips`；对「开始」有**首字过长**修正（见同文件内 `_t0s_refine_if_leading_stretched`），用于缓解字级在段末把「开」拉成数秒、导致第三段**片头发空**的问题。

## 推荐命令（有公网音轨 URL）

在 **`umi_scripts_csw`** 目录下：

```bash
python3 nano_sync/segment_by_record_markers.py "/path/to/video.mp4" \
  --file-url "https://…/track.wav" \
  --enable-words \
  --out-json /tmp/seg_out.json \
  --cut-dir /path/to/out_clips
```

- **`--include-stop-mouth-sec`**：默认在脚本中约为 `0.12`；`0` 为关闭。与「停止句末」`min` 限幅在 `pair_record_clips` 中完成。
- **`--from-json`**：不调用 API，直接读**已保存的**完整 verbose JSON（需含 `segments`；最好含 `_filetrans_char_timeline` 以便走字级重排）。

**保存整份 ASR 便于复跑（不重复打 API）**：

```bash
python3 nano_sync/segment_by_record_markers.py "/path/to/video.mp4" \
  --file-url "https://…/track.wav" \
  --enable-words \
  --asr-only \
  --out-json /tmp/asr_full.json
```

## 公网 `file_url` 的实践

- **推荐**：`nano_sync/upload_wav_oss.py` 上传至自有 OSS/可读桶，得到稳定直链（需 `pip install oss2` 与 OSS 环境变量）。
- **临时方案**：本机曾用 uguu 等临时直链作调试；**链路易失效**，只宜一次性试验。

## 输出位置约定（示例）

- 切片目录：建议用仓库**外**路径，或本目录下 `out_clips/`（已 `.gitignore` 忽略 `out_*/`），例如 `clip_0000.mp4 …`。
- 汇总 JSON：含 `duration_sec`、`marker_defaults`（含 `include_stop_mouth_sec`）、`clips[]` 各时间戳与 `warnings`。

## 已知注意点

1. **转写 `duration` 与容器时长**：`verbose_json` 里的 `duration` 可能来自 ASR 的语音结束估计，**不一定**与 `ffprobe` 片长完全一致；`pair_record_clips` 会用传入的 `duration_sec` 做上界。若只关心与 ASR 对齐的切点，以片段时间列为准；若与 EOF 强相关，可用 ffprobe 核对。
2. **字级异常**：除「开」被拉长外，条尾「停」等也可能出现**单字时宽异常**；本流程对**开始句**有首字过长向后的修正，**停止句**仍主要取「停」的句起点 `t0`（不向前误挪到片尾点质量）。
3. **密钥与临时 URL**：`DASHSCOPE_API_KEY` 与只读直链不要写入仓库；临时上传文件注意隐私与有效期。

## 本仓库内相关文件

| 文件 | 作用 |
|------|------|
| `segment_by_record_markers.py` | CLI：ASR、字级重排、配对、`--cut-dir` 裁片 |
| `record_marker_segments.py` | 字级行展开、口令 span、配对、口型余量、首字过长修正 |
| `qwen_filetrans_asr.py` | 百炼 filetrans 提交/轮询、转 whisper 形、字级行 `_filetrans_char_timeline` |
| `upload_wav_oss.py` | 上传至 OSS 输出公网 **HTTPS** 供 `file_url` |
| `audio_extract.py`（`umi_scripts_csw`） | 抽轨、`find_ffmpeg` |
| `test_record_marker_segments.py` | 单元测试（不联网） |
| `test_qwen_filetrans_normalize.py` | 离线单测：transcription→segments 映射 |
| `record_marker_phrases.py` | 默认口令与关键词，与 TTS/分段脚本共用 |
| `record_marker_tts_listener.py` | 采集中 TTS 播报（与分段后置处理独立；见 umi README） |
| `__init__.py` | 包入口 |

---

*文档随实现变更时请同步更新本文件。*

## 替代方案：双标定音（不依赖 ASR）

见 [`../nano_sync_freq/README.md`](../nano_sync_freq/README.md) —— 采集中播两种 chirp、后期互相关切分，与上文的百炼/口语时间轴**独立**。
