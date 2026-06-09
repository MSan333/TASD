# GRPO Ranking 训练指南

## 目录

- [环境准备](#环境准备)
- [数据准备](#数据准备)
- [训练方式](#训练方式)
  - [方式 A: 本地训练](#方式-a-本地训练)
  - [方式 B: 星云提交](#方式-b-星云提交)
- [超参配置](#超参配置)
- [日志与监控](#日志与监控)
- [常见问题](#常见问题)

---

## 环境准备

### 1. Conda 环境

本项目依赖 `python3.10` conda 环境，需要包含以下核心包：

| 包名 | 最低版本 | 用途 |
|------|---------|------|
| torch | 2.7+ | 训练框架 |
| ray | 2.41+ | 分布式调度 |
| vllm | 0.10+ | 推理引擎 (rollout) |
| swanlab | 0.8+ | 日志追踪 |
| verl | 0.7+ | RL 训练框架 |

```bash
# 激活环境
conda activate python3.10

# 验证依赖
python -c "import torch, ray, vllm, swanlab; print('OK')"

# 安装本项目 (editable mode)
pip install -e . --no-deps --no-build-isolation
```

### 2. SwanLab 登录

训练日志通过 SwanLab Cloud 上报，首次使用需登录：

```bash
python -c "import swanlab; swanlab.login('o4MGQAOSX8rGztH69Jj5P')"
```

登录成功后会显示 `Currently logged in as: 三三`。

如果 API Key 失效，请到 https://swanlab.cn/settings 重新获取，并更新 `exp/grpo/env.sh` 中的 `SWANLAB_API_KEY`。

### 3. 星云账号（仅星云提交需要）

提交星云任务需要以下 3 个环境变量，建议写入 `~/.bashrc`：

```bash
export OPENLM_TOKEN="your_openlm_token"
export OSS_ACCESS_ID="your_oss_access_id"
export OSS_ACCESS_KEY="your_oss_access_key"
```

---

## 数据准备

### 自动下载

```bash
conda activate python3.10
python -m exp.grpo.download_data
```

默认下载到 `exp/grpo/data/grpo_v3/`，包含：
- `train.parquet` — 训练集
- `test.parquet` — 测试集 (可选)
- `tipbank/` — TipBank 知识库

### 手动下载 (OSS)

```bash
# 下载 ranking 数据
mkdir -p exp/grpo/data/ranking
ossutil64 cp oss://lazada-ai-model/ad/guoshauile.gsl/data/ranking/train.parquet exp/grpo/data/ranking/
ossutil64 cp oss://lazada-ai-model/ad/guoshauile.gsl/data/ranking/test.parquet exp/grpo/data/ranking/
```

### 模型准备

本地训练需要模型权重，默认路径 `/home/guoshuaile.gsl/models/qwen3-8b`：

```bash
# 如果需要从 OSS 下载
mkdir -p ~/models/qwen3-8b
ossutil64 cp -r oss://lazada-ai-model/ad/guoshauile.gsl/model/qwen3-8b/ ~/models/qwen3-8b/
```

---

## 训练方式

### 方式 A: 本地训练

适用于有 GPU 的开发机。脚本会自动检测 GPU 数量并适配 batch size。

```bash
cd /path/to/TASD
conda activate python3.10

# 快速启动（使用默认配置）
bash run_local_grpo_ranking.sh

# 或先加载环境变量再启动
source exp/grpo/env.sh
bash run_local_grpo_ranking.sh

# 后台运行
nohup bash run_local_grpo_ranking.sh > outputs/training.log 2>&1 &
tail -f outputs/training.log
```

#### 指定 GPU

```bash
# 指定 GPU 数量
N_GPUS=4 bash run_local_grpo_ranking.sh

# 指定哪些 GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_local_grpo_ranking.sh

# 单卡
CUDA_VISIBLE_DEVICES=0 bash run_local_grpo_ranking.sh
```

#### 覆盖超参

```bash
# 改学习率
LR=5e-6 bash run_local_grpo_ranking.sh

# 改多个超参
LR=5e-6 TRAIN_BATCH_SIZE=16 KL_COEF=0.01 bash run_local_grpo_ranking.sh

# 改模型路径
MODEL_PATH=/path/to/another/model bash run_local_grpo_ranking.sh
```

### 方式 B: 星云提交

适用于使用 Nebula 集群训练（4 卡 H20）。

```bash
cd /path/to/TASD

# 1. 加载环境变量
source exp/grpo/env.sh

# 2. 提交任务（1 个节点 = 4 卡）
bash nebula_scripts/submit_job.sh \
    nebula_scripts/grpo/grpo_ranking_gsl.sh \
    1 \
    lazada_llm_ad_h20
```

#### 提交前检查清单

- [ ] `OPENLM_TOKEN` 已设置
- [ ] `OSS_ACCESS_ID` 已设置
- [ ] `OSS_ACCESS_KEY` 已设置
- [ ] 模型已上传到 OSS: `oss://lazada-ai-model/ad/guoshauile.gsl/model/qwen3-8b`
- [ ] 数据已上传到 OSS: `oss://lazada-ai-model/ad/guoshauile.gsl/data/ranking/`

#### 查看任务状态

提交后终端会输出任务名，可通过 Nebula 控制台查看状态和日志。

---

## 超参配置

### 核心超参

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DATASET` | `ranking` | 数据集名称 |
| `LR` | `1e-5` | 学习率 |
| `KL_COEF` | `0.05` | KL 散度系数 |
| `TOTAL_TRAINING_STEPS` | `500` | 总训练步数 |
| `MODEL_PATH` | (见 env.sh) | 基底模型路径 |

### GPU 自适应超参

`run_local_grpo_ranking.sh` 会根据 GPU 数量自动选择以下配置：

| GPU | train_batch | mini_batch | rollout_n | val_n |
|-----|-------------|------------|-----------|-------|
| 1 卡 | 8 | 2 | 4 | 4 |
| 2 卡 | 16 | 4 | 4 | 8 |
| 4 卡 | 32 | 8 | 8 | 16 |

> 如果通过 `source env.sh` 或环境变量预设了这些值，脚本会优先使用预设值。

### Reward 架构

训练使用两层 Reward 架构：

1. **L0: Format Gate** — 格式校验，非法输出返回 -2.0
2. **L2: Knowledge-Grounded Judge** — TipBank 检索 + 意图识别 + 知识对比打分

LLM Judge 使用 DashScope API (qwen-plus)，配置通过以下环境变量控制：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `OPENAI_API_KEY` | (见 env.sh) | DashScope API Key |
| `OPENAI_BASE_URL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | API 端点 |
| `OPENAI_MODEL` | `qwen3.6-max-preview` | Judge 使用的模型 |

---

## 日志与监控

### SwanLab (推荐)

训练指标自动上传到 SwanLab Cloud：https://swanlab.cn

- **模式**: cloud（实时上传）
- **项目**: GRPO-Ranking
- **指标**: reward, KL, loss, format_valid_rate 等

### 本地日志

```bash
# 查看实时日志
tail -f outputs/grpo_ranking_local_training.log

# 查看 SwanLab 本地缓存
ls outputs/grpo_ranking_local/swanlab_logs/
```

### WandB (备用)

默认关闭 (`WANDB_MODE=offline`)。如需启用：

```bash
export WANDB_MODE=online
export WANDB_API_KEY="your_wandb_key"
```

---

## 常见问题

### Q: `ModuleNotFoundError: No module named 'ray'`

**原因**: 使用了错误的 Python 环境。

**解决**: 确保激活 conda python3.10 环境：
```bash
conda activate python3.10
# 或确保 PATH 中包含:
export PATH="/opt/conda/envs/python3.10/bin:${PATH}"
```

### Q: SwanLab 登录失败

**解决**:
```bash
# 重新登录
python -c "import swanlab; swanlab.login('o4MGQAOSX8rGztH69Jj5P', relogin=True)"
```

如果 API Key 失效，请到 https://swanlab.cn/settings 获取新 Key。

### Q: 训练数据不存在

**解决**:
```bash
# 方式 1: 自动下载
python -m exp.grpo.download_data

# 方式 2: 手动下载
mkdir -p exp/grpo/data/ranking
ossutil64 cp oss://lazada-ai-model/ad/guoshauile.gsl/data/ranking/train.parquet exp/grpo/data/ranking/
ossutil64 cp oss://lazada-ai-model/ad/guoshauile.gsl/data/ranking/test.parquet exp/grpo/data/ranking/
```

### Q: 星云提交报错 `OPENLM_TOKEN not set`

**解决**: 设置 Nebula 账号环境变量：
```bash
export OPENLM_TOKEN="your_token"
export OSS_ACCESS_ID="your_id"
export OSS_ACCESS_KEY="your_key"
```

建议写入 `~/.bashrc` 持久化。

### Q: GPU 显存不足 (OOM)

**解决**: 降低 batch size 或 GPU 显存利用率：
```bash
MINI_BATCH_SIZE=1 TRAIN_BATCH_SIZE=4 ROLLOUT_N=2 GPU_MEM_UTIL=0.4 bash run_local_grpo_ranking.sh
```

### Q: 如何修改 Reward 函数?

Reward 代码位于 `exp/grpo/` 目录：
- `reward_fn.py` — 主 reward 计算逻辑
- `ranking.py` — 单条 reward 适配器
- `ranking_batch.py` — 批量 reward (含 async LLM 调用)
- `knowledge_judge.py` — Knowledge Judge 评分
- `intent_recognizer.py` — 意图识别
- `tipbank_loader.py` — TipBank 加载和检索

---

## 文件结构

```
TASD/
├── exp/grpo/
│   ├── env.sh                  # 环境变量配置 (本文件)
│   ├── TRAINING_GUIDE.md       # 训练教程 (本文件)
│   ├── data/ranking/           # 训练数据 (需下载, 不提交 git)
│   │   ├── train.parquet
│   │   └── test.parquet
│   ├── reward_fn.py            # Reward 计算
│   ├── ranking.py              # 单条 reward 适配器
│   ├── ranking_batch.py        # 批量 reward
│   ├── knowledge_judge.py      # Knowledge Judge
│   ├── intent_recognizer.py    # 意图识别
│   ├── tipbank_loader.py       # TipBank 加载
│   └── download_data.py        # 数据下载脚本
├── run_local_grpo_ranking.sh   # 本地训练脚本
├── nebula_scripts/grpo/
│   └── grpo_ranking_gsl.sh     # 星云训练脚本
└── verl/                       # veRL 训练框架
```
