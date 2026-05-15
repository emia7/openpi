# 双音标定（nano_sync_freq）— 快速上手

采集中用 **a / c** 在耳机里播 **chirp 标音** 打点；录完整段长视频后，在本地用检测找起停时间、再 `ffmpeg` **流复制** 切成多段。**全程可离线。**

本文档是 **nano_sync_freq** 从标音 assets → 长录 → 首切 → 浏览器改表 → 重切 的**固定流程**；后文 **§4 稳定约定** 与 `annotate/app.js` 须保持一致，改代码时同步改文档。

在仓库里进 **`umi_scripts_csw`** 再执行下述命令（把 `cd` 换成你本机路径）：

```bash
cd /path/to/umi_scripts_csw
```

依赖、脚本参数、门槛调参见同目录 **README.md**、**config.py**。本文只包「能从头切完」的固定顺序。

---

## 1. 一次性：生成标音 WAV

改动了 `config.py` 里与**标音波形**相关的内容后，**必须**重跑：

```bash
python3 nano_sync_freq/build_assets.py
```

得到 `nano_sync_freq/assets/start.wav`、`stop.wav`。**采集中**与**后处理**检测只能用**同一次**生成结果。

---

## 2. 采集中：a 开段、c 停段 录长片

```bash
python3 nano_sync_freq/pedal_freq_listener.py
# 需要不抢该终端焦点时（要 pynput，Linux Wayland 常不行）:
python3 nano_sync_freq/pedal_freq_listener.py --global
```

- **a** = 开一段、**c** = 停一段，与后文标注页默认快捷键一致。  
- **先**让运行该脚本的**终端窗口**获焦，再按 a / c。  
- 先确认能听见标音：`python3 nano_sync_freq/play_markers.py --self-test`（或 `-i` 试听单文件）。  

录完得到如 **`record.mp4`**。后面算法按「**每段 = 一个起 后接 第一个停**」成对、切片。

---

## 3. 首跑：找峰 + 成对 + 切出 clip

将路径换成你的**原片**和输出目录（`out_freq` 仅为示例）：

```bash
python3 nano_sync_freq/segment_by_freq_markers.py /path/to/record.mp4 \
  --cut-dir ./out_freq \
  --markers-out ./out_freq/markers.json
```

得到：

- **`out_freq/clip_0000.mp4` …** — 片段。  
- **`out_freq/markers.json`** — 全部分析与起停表，**要留档**（给标注、重切）。

**可选项（按需加在一条命令上）**：

- 只出 JSON 不切：`--write-markers-only` + `--markers-out`；或不要 `--cut-dir`。  
- 不要「停音」后面的**尾长**（多录进一截停 chirp）：`--no-stop-tail`。  
- 自订尾长秒数：`--stop-tail-sec 0.12` 或看 `config.py` 里 `INCLUDE_STOP_BEEP_TAIL_SEC`。

---

## 4. （建议）浏览器里对时间、改起停、导出

首跑不准时，在本地页里听原片、对表格，再导出**改过的起停表**。

**推荐**起小服务（`--video`、`--markers` 换成**同一原片**与**第 3 步的** `markers.json`）：

```bash
python3 nano_sync_freq/annotate_server.py \
  --video /path/to/record.mp4 \
  --markers /path/to/out_freq/markers.json \
  --port 8765
```

浏览器打开终端里给的地址（如 **`http://127.0.0.1:8765/`**；端口被占用时换一个 `--port` 即可）。**只本机用。**

- 不启服务时：在页面里**选文件**加载 `markers.json` 与视频。  
- 时间轴有**只读系统层**与**可编辑**起停；**滚轮 / 按钮**可缩放、平移。  
- 在**起/停时间输入框外**（才响应打标/全局撤销）：**a / c** 在播放头加起/停，**Space** 播/停；**单击**可编辑绿/红竖线选中，再按 **e** 把**该时刻**在「起表」与「停表」之间**挪动**（误标成起/停时用；只改这一条，不与其他点对换）；**⌘/Ctrl+Z** 撤销、**⇧+⌘/Ctrl+Z** 或 **Ctrl+Y** 重做；在**输入框里**仍用**系统自带**的文字撤销。  
- 点 **「导出 `edited_markers.json`」** 存到如 `~/Downloads/`。

### 稳定约定：标注页（`nano_sync_freq/annotate/`）

**以后均按此语义实现/使用**；若改 `app.js` 行为，**必须**同步更新本小节与 `annotate/index.html` 顶栏说明。

| 操作 | 行为（与 `pair_markers` / 切片**独立**） |
|------|----------------------------------------|
| **a** | 在**当前播放头**时间写入可编辑 **「起」**表（`start_times`），再按时间排序。 |
| **c** | 同上写入 **「停」**表（`stop_times`），排序。 |
| **Space** | 视频播放 / 暂停。 |
| **单击**时间轴上绿（起）或红（停）竖线 | **选中**该条（高亮）；点在空白处取消选中。 |
| **e** | **须先选中一根竖线**。把该条对应的**唯一时刻**从起表**删掉并插入停表**，或从停表**移到起表**；然后两表各自按时间排序。**不**与其它点对换数值，**不**按贪心配对去换另一根柱。**实现**：`annotate/app.js` 内 `reclassifyPickAtSelection()`。 |
| **⌘/Ctrl+Z** | 撤销对起停表的修改（**输入框外**；约 **120** 步历史）。 |
| **⇧+⌘/Ctrl+Z** 或 **Ctrl+Y** | 重做。 |
| 表内**输入框** | 使用**浏览器自带**文字撤销，不走上两项「整表」撤销。 |
| 加载新 `markers.json` | 清空整表撤销 / 重做栈。 |

**成片、预览条**上的「起接第一个停」仍由 **`segment_by_freq_markers.py` 与 `pair_markers` 的贪心规则**决定，与 **e** 的「单点改类」无关；导出 JSON 后重切时再用该规则成对。

---

## 5. 用改表重切（不再跑 NCC 检测）

**`record.mp4`** 要仍是同一条；JSON 用导出的（里面一般有 `file` 等字段）：

```bash
python3 nano_sync_freq/segment_by_freq_markers.py /path/to/record.mp4 \
  --markers-in /path/to/edited_markers.json \
  --cut-dir /path/to/out_revised \
  --markers-out /path/to/out_revised/markers.json
```

- `--cut-dir` 用**新目录**，避免冲掉未备份的旧片。  
- 这里只做**成对 + 片尾 + 切**，不重新做互相关。  
- 没改起停、只想**改尾长**时，也可把**首跑**的 `markers.json` 作 `--markers-in`，并加 `--no-stop-tail` 或 `--stop-tail-sec …`。

---

## 6. 验收

- 播放器按序听 `clip_000*.mp4`，看段数、起止与听感。  
- 有开发环境时跑自测：

```bash
python3 -m pytest nano_sync_freq/tests/ -q
```

段数/门槛不对：改 `config`、必要时重跑 **1 → 3**，再视情况走 **4 → 5**。
