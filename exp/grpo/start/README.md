# GRPO Ranking 快速开始

## 环境要求

- **Python**: 3.10+
- **GPU**: NVIDIA H20 / A100 / H100（至少 1 张）
- **CUDA**: 12.x
- **磁盘**: ~30GB（模型 + 虚拟环境）

---

## 🚀 一键上手

### Step 1: 安装环境（uv 管理）

```bash
bash exp/grpo/start/setup.sh
```

这会自动：
- 安装 [uv](https://docs.astral.sh/uv/)（Rust 实现的 Python 包管理器，比 pip 快 10-100x）
- 创建 `.venv/` 虚拟环境（Python 3.10）
- 安装 PyTorch（CUDA 12.4）+ 所有训练依赖
- 以开发模式安装 verl

### Step 2: 本地训练

```bash
bash exp/grpo/start/train.sh
```

自动检测 GPU 数量并调整 batch size。

### Step 3: 提交星云

```bash
bash exp/grpo/start/submit_nebula.sh
```

密钥已配置在 `submit_nebula.sh` 中（本地文件，不会提交到 git）。

---

## 配置说明

### 密钥配置

所有密钥配置在 `exp/grpo/start/submit_nebula.sh` 中，需要以下 4 个：

| 密钥 | 说明 | 获取方式 |
|------|------|---------|
| `OPENLM_TOKEN` | Nebula 平台认证 | [OpenLM 控制台](https://openlm.alibaba-inc.com) |
| `OSS_ACCESS_ID` | 阿里云 OSS Access Key ID | [RAM 控制台](https://ram.console.aliyun.com) |
| `OSS_ACCESS_KEY` | 阿里云 OSS Access Key Secret | 同上 |
| `SWANLAB_API_KEY` | SwanLab 实验追踪 | [SwanLab 设置](https://swanlab.cn) |

### 超参覆盖

所有超参都有默认值，可通过环境变量覆盖：

```bash
# 本地训练 — 环境变量覆盖
LR=5e-6 TOTAL_TRAINING_STEPS=100 bash exp/grpo/start/train.sh

# 指定 GPU
CUDA_VISIBLE_DEVICES=0,1 N_GPUS=2 bash exp/grpo/start/train.sh

# 星云提交 — KEY=VALUE 参数覆盖
bash exp/grpo/start/submit_nebula.sh LR=5e-6 TRAIN_BATCH_SIZE=64

# 多节点
WORLD_SIZE=2 bash exp/grpo/start/submit_nebula.sh
```

### 默认超参

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `DATASET` | `ranking` | 数据集名 |
| `LR` | `1e-5` | 学习率 |
| `MINI_BATCH_SIZE` | 按GPU自适应 | PPO mini-batch |
| `TRAIN_BATCH_SIZE` | 按GPU自适应 | 训练 batch |
| `ROLLOUT_N` | 按GPU自适应 | 每 prompt 采样数 |
| `MODEL_PATH` | `/home/guoshuaile.gsl/models/qwen3-8b` | 基底模型 |
| `KL_COEF` | `0.05` | KL 散度系数 |
| `TOTAL_TRAINING_STEPS` | `500` | 总训练步数 |

### GPU 自适应配置

| GPU 数量 | train_batch | mini_batch | rollout_n | val_n |
|----------|------------|------------|-----------|-------|
| 1 | 8 | 2 | 4 | 4 |
| 2 | 16 | 4 | 4 | 8 |
| 4+ | 32 | 8 | 8 | 16 |

---

## 文件结构

```
exp/grpo/start/
├── setup.sh           # 一键安装环境（uv）
├── train.sh           # 本地训练
├── submit_nebula.sh   # 星云提交（含密钥，不入 git）
└── README.md          # 本文档

exp/grpo/data/         # 训练数据（.gitignore 忽略）
├── ranking/
│   ├── train.parquet
│   └── test.parquet
```

---

## 常见问题

### Q: `uv: command not found`

手动安装：
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

### Q: 本地已有 conda 环境，还需要 uv 吗？

不需要。如果 notebook 里已有 `python3.10.13` conda 环境，`train.sh` 会自动检测并使用它，优先级：`.venv/`（uv）> `python3.10.13` > `python3.10`。

### Q: `HFValidationError: Repo id must be in the form 'repo_name'`

已修复。`verl/utils/tokenizer.py` 会自动对本地路径设置 `local_files_only=True`。

### Q: notebook 只有 1 张 GPU，能跑吗？

可以。`train.sh` 会自动检测 GPU 数量并调小 batch size。
