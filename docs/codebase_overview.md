# openpi 代码库概览

本文档从**仓库布局、脚本与模块**角度说明 openpi 实现了什么、各部分如何协作。更详细的安装、命令行用法与排错仍以根目录 [README.md](../README.md) 为准。

## 项目定位

**openpi** 是 Physical Intelligence 开源的机器人视觉-语言-动作（VLA）相关代码与模型发布仓库，提供：

- **π₀**：基于流匹配（flow matching）的 VLA。
- **π₀-FAST**：基于 FAST 动作 tokenizer 的自回归 VLA。
- **π₀.₅**：在 π₀ 基础上的升级版本（本仓库中训练与推理侧对 π₀.₅ 主要支持 **flow matching 头**）。

对上述模型，仓库提供在 **1 万小时以上** 机器人数据上预训练的 **base 权重**，以及面向 ALOHA、DROID、LIBERO 等场景的 **微调 / 可直接推理** 的检查点（默认从 `gs://openpi-assets` 拉取并缓存在 `~/.cache/openpi`，可通过 `OPENPI_DATA_HOME` 覆盖）。

实现上同时支持 **JAX / Flax**（训练主路径）与 **PyTorch**（π₀、π₀.₅ 的推理与微调；部分 JAX 侧能力在 PyTorch 中尚未对齐，见 README「PyTorch Support」）。

---

## 核心能力一览

| 能力 | 说明 |
|------|------|
| **模型与前向** | `src/openpi/models/`：π₀、π₀-FAST、π₀.₅ 相关结构；Gemma / SigLIP / tokenizer 等组件；`models_pytorch/` 为 PyTorch 实现及 transformers 补丁文件。 |
| **数据与变换** | `src/openpi/transforms.py` 与各 `DataConfig`：把 LeRobot（及 RLDS 等）数据映射为模型输入；训练前需 **归一化统计**。 |
| **训练（JAX）** | `scripts/train.py`：加载权重、数据管线、优化器、FSDP 分片、检查点与 W&B 等。 |
| **训练（PyTorch）** | `scripts/train_pytorch.py` + `torchrun`：单卡/多卡 DDP、多节点等。 |
| **归一化统计** | `scripts/compute_norm_stats.py`（及仓库内若干变体脚本）：按 config 扫数据集，写出 `norm_stats.json`；详见 [norm_stats.md](norm_stats.md)。 |
| **策略与推理** | `src/openpi/policies/`：`Policy` 封装、`policy_config.create_trained_policy` 从 checkpoint 构建可调用策略；各环境有专用 Input/Output 映射（如 `droid_policy`、`aloha_policy`、`libero_policy`）。本 fork 还可能包含面向特定机械臂/表示的扩展 policy（如 Franka、UMI、XV 等系列）。 |
| **服务化推理** | `scripts/serve_policy.py`：基于 WebSocket  hosting 策略；`openpi-client` 提供远端客户端与轻量 runtime。 |
| **权重与下载** | `src/openpi/shared/download.py`、`scripts/download_checkpoint.py` 等：从 GCS 等拉取检查点。 |
| **JAX → PyTorch** | `examples/convert_jax_model_to_pytorch.py`：格式转换后在 PyTorch 路径下使用同一套 Policy API。 |

---

## 目录与模块说明

### `src/openpi/`（主 Python 包）

- **`models/`**  
  VLA 模型定义、配置（如 `pi0_config`）、LoRA、与预训练骨干相关的模块。

- **`models_pytorch/`**  
  PyTorch 版 π₀ / π₀.₅ 与预处理；`transformers_replace/` 为需覆盖到已安装 `transformers` 的补丁（README 中有说明与风险提示）。

- **`policies/`**  
  训练与推理共用的「观测 / 动作空间 ↔ 模型张量」约定；`policy_config.py` 负责按 `TrainConfig` + checkpoint 目录装配策略。

- **`training/`**  
  - `config.py`：**单一事实来源**级别的训练配置注册（数据集、变换、超参、权重加载器等）。  
  - `data_loader.py`、`droid_rlds_dataset.py`：数据加载与 DROID RLDS 相关逻辑。  
  - `checkpoints.py`、`optimizer.py`、`sharding.py`、`weight_loaders.py`：检查点、优化、分片与权重初始化。

- **`serving/`**  
  WebSocket 策略服务实现（可与 `scripts/serve_policy.py` 及 `serve_policy_csw.py` 等入口配合）。

- **`shared/`**  
  数组类型、归一化、图像工具、NNX 辅助、从云端下载资源等通用能力。

- **`transforms.py`**  
  可组合的数据变换，贯穿 norm stats 计算与训练管线。

### `scripts/`（常用入口）

| 脚本 | 作用 |
|------|------|
| `train.py` | JAX 训练主入口（`uv run scripts/train.py <config> ...`）。 |
| `train_pytorch.py` | PyTorch 训练入口。 |
| `compute_norm_stats.py` | 为指定 config 计算并保存归一化统计（训练前通常必跑）。 |
| `compute_norm_stats_*.py` | 针对特定动作维或带图像等场景的变体（按名称区分用途）。 |
| `serve_policy.py` | 启动策略 WebSocket 服务（checkpoint 或内置默认环境策略）。 |
| `serve_policy_csw.py` | 与扩展 serving 实现相关的变体入口（若你使用 CSW 路径，以脚本内说明为准）。 |
| `download_checkpoint.py` | 检查点下载辅助。 |
| `train_test.py` | 训练相关测试/冒烟用途（以文件内逻辑为准）。 |

### `examples/`

面向具体机器人或基准的 **可运行示例** 与文档：

- **LIBERO**：数据转换、Docker 化评测、与 `serve_policy` 联调。  
- **DROID**：实机/管线说明、全量 DROID 训练指引（`README_train.md`）、数据转换与 idle 过滤等。  
- **ALOHA**：仿真（`aloha_sim`）与真机（`aloha_real`）。  
- **UR5**：UR5 相关说明与脚本。  
- **simple_client**：无机器人时随机观测跑通推理。  
- **inference.ipynb** / **policy_records.ipynb**：笔记本形式试用与记录。

### `packages/openpi-client`

独立子包（workspace 成员），供 **机器人端或远端轻环境** 使用：

- WebSocket 客户端策略（`websocket_client_policy`）、msgpack+numpy 序列化。  
- `runtime/`：环境、订阅者与 `PolicyAgent` 等抽象，便于把「连策略服务」接进控制循环。

### `docs/`

- [docker.md](docker.md)：Docker 安装与运行。  
- [remote_inference.md](remote_inference.md)：远端推理架构与接入方式。  
- [norm_stats.md](norm_stats.md)：归一化统计与预训练统计复用。  
- **本文**：[codebase_overview.md](codebase_overview.md) 代码库功能与结构总览。

---

## 典型工作流（与脚本对应）

1. **仅推理（本机）**  
   使用 `policy_config.create_trained_policy` + `policy.infer`（见 README 代码片段），或跑 `examples/simple_client`、`examples/*/main.py`。

2. **远端推理**  
   服务端：`scripts/serve_policy.py`；客户端：`openpi-client` 与 [remote_inference.md](remote_inference.md)。

3. **在自己的数据上微调（JAX）**  
   数据进 LeRobot → 在 `training/config.py` 与对应 `policies/*_policy.py` 定义映射 → `compute_norm_stats.py` → `train.py` → `serve_policy.py` 或评测脚本。

4. **PyTorch 路径**  
   可选 `convert_jax_model_to_pytorch.py` → 配置中指定 PyTorch 权重路径 → `train_pytorch.py` / 同一套 `create_trained_policy` 推理。

---

## 依赖与环境要点（简）

- Python **≥ 3.11**，依赖由 **uv** 管理（`pyproject.toml`）。  
- 训练侧以 **JAX CUDA** 为主；PyTorch 固定版本与 `transformers` 补丁版本见 README。  
- 数据侧深度集成 **LeRobot**；DROID 全量训练等场景可能需要 **rlds** 可选依赖组（TensorFlow CPU、dlimp 等）。

---

## 小结

openpi 将 **VLA 模型、数据管线、训练、检查点分发、策略封装与 WebSocket 服务** 连成一条完整链路，并通过 **examples** 与 **openpi-client** 覆盖从桌面试验到机载/远端部署的常见形态。若你只关心「该跑哪条命令」，请优先阅读 [README.md](../README.md) 中对应平台章节；若关心「代码在哪一层」，可按上表在 `src/openpi`、`scripts`、`examples` 中定位。
