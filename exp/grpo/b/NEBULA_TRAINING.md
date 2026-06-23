# GRPO v7 星云训练指南

## 版本说明

**v7 (DAPO Filter + Top-3 Precision)** 相比 v4 的主要改进：

| 问题 | v4 | v7 |
|------|-----|-----|
| KL 正则化 | `use_kl_loss=False`，KL 从 0.003 爆炸到 0.7 | `use_kl_loss=True, kl_loss_coef=0.01` |
| 格式惩罚 | 加法惩罚 `-0.5`，占 reward 64-94% | DAPO filter: `invalid=0`，GRPO 自动产生负 advantage |
| 排序质量 | 所有位置都有信号，归一化混乱 | Top-3 Precision，只看前 3 位置，归一化到 `[0,1]` |
| Advantage 归一化 | `norm_adv_by_std=False`（错误） | `norm_adv_by_std=True`（GRPO 标准做法） |
| Rollout IS | `rollout_is=token`，策略漂移后变成噪声 | 去掉 rollout_is |
| Reward 公式 | `0.6×rule + 0.4×judge`（judge 未归一化） | `0.7×top3 + 0.3×judge_norm`（judge 归一化到 `[0,1]`） |

**论文依据**：DAPO (ByteDance 2025)、DeepSeek-R1、PRM (Lightman 2023)

---

## 密钥准备

提交脚本中 `SWANLAB_API_KEY` 已明文写入（无安全风险），以下 3 个敏感密钥需要通过环境变量传入。使用前请在本地设置（建议写入 `~/.bashrc`）：

```bash
# OpenLM Token（用于模型加载）
export OPENLM_TOKEN="your_openlm_token"

# OSS 访问凭证（用于数据/模型/结果存储）
export OSS_ACCESS_ID="your_oss_access_id"
export OSS_ACCESS_KEY="your_oss_access_key"
```

> ⚠️ **注意**：这 3 个密钥不要提交到 git 仓库中。如果 `submit_nebula_uv.sh` 需要提交到 git，请确保其中不包含这些敏感信息（当前脚本已从环境变量读取）。

---

## 星云提交流程

### 1. 前置条件

**本地环境**（notebook 或开发机）：
- `nebulactl` CLI 已安装
- git 已配置，能 push 到远端
- 代码已 commit 并 push 到 `origin`
- 上述密钥环境变量已设置

**星云环境**：

| 方式 | 镜像 | 说明 |
|------|------|------|
| 方式 A（默认） | `hub.docker.alibaba-inc.com/mdl/notebook_saved:guoshuaile.gsl_33v2_20260609193943` | 预装 torch/vllm/ray，训练时 pip install 依赖 |
| 方式 B（自定义） | **需自行构建并推送** | 预装 uv + 项目依赖，训练时直接激活环境 |

**方式 B 自定义镜像配置**：
- 在 `submit_nebula_uv.sh` 中通过 `CUSTOM_DOCKER_IMAGE` 环境变量指定镜像名称
- 镜像需基于星云 notebook_saved 镜像，预装 uv 和项目依赖
- 构建推送后，将完整镜像 tag 填入提交命令即可

### 2. 提交任务

#### 方式 A：使用默认镜像 + pip 安装依赖

```bash
cd /path/to/TASD

# 默认：单节点 4 卡，500 steps
bash exp/grpo/start/submit_nebula.sh

# 覆盖超参
bash exp/grpo/start/submit_nebula.sh LR=5e-6 TOTAL_TRAINING_STEPS=200

# 多节点（2 节点 × 4 卡 = 8 卡）
WORLD_SIZE=2 bash exp/grpo/start/submit_nebula.sh
```

#### 方式 B：使用自定义镜像 + uv 环境（推荐）

适用于已将本地环境打包为 Docker 镜像的场景。镜像中需预装 uv 和项目依赖，训练时直接激活环境运行，无需每次 `pip install`。

```bash
cd /path/to/TASD

# 提交（必须指定 CUSTOM_DOCKER_IMAGE）
CUSTOM_DOCKER_IMAGE=<your_image_tag> bash exp/grpo/b/submit_nebula_uv.sh

# 覆盖超参
CUSTOM_DOCKER_IMAGE=<your_image_tag> bash exp/grpo/b/submit_nebula_uv.sh LR=5e-6 TOTAL_TRAINING_STEPS=200

# 多节点
CUSTOM_DOCKER_IMAGE=<your_image_tag> WORLD_SIZE=2 bash exp/grpo/b/submit_nebula_uv.sh
```

**自定义镜像要求**：
- 基于星云 notebook_saved 镜像（已有 conda python3.10.13 + torch + vllm + ray）
- 安装了 **uv**
- 项目依赖已通过 `uv pip install --system -e .` 预装（或镜像中有 `.venv`）

**对应的训练脚本**：`exp/grpo/b/grpo_ranking_gsl_v7_uv.sh`
- 使用 uv 激活环境（优先 `.venv`，回退到系统 Python）
- 去掉了 `pip install` 步骤（镜像中已预装）
- 超参：`TRAIN_BATCH_SIZE=8, ROLLOUT_N=8, MINI_BATCH_SIZE=4`

### 3. 查看日志

提交成功后会打印 `task_id`：
```
✅ 任务已提交
Task ID: abc123def456
Log URL: https://nebula2.alibaba-inc.com/log/logview/xdl/view?task_id=abc123def456
```

**日志页面**：
- 点击链接打开 Nebula LogView
- 选择 `worker-0` 查看主节点日志
- 关键指标：`critic/score/mean`, `rollout_corr/kl`, `actor/entropy`, `reward/*/mean`

### 4. SwanLab 监控

训练会自动记录到 SwanLab（cloud 模式）：
- **API Key**: `o4MGQAOSX8rGztH69Jj5P`（已配置在脚本中）
- **Project**: `GRPO-Ranking`
- **Experiment**: `grpo_ranking_v7_YYYYMMDD_HHMMSS`

**关注的指标**：
1. `critic/score/mean` - 整体 reward（应该逐步上升）
2. `reward/rule_score/mean` - Top-3 Precision（核心指标，应该上升）
3. `rollout_corr/kl` - KL 散度（应该 < 0.1，v4 爆炸到 0.7）
4. `actor/entropy` - 策略熵（应该稳定在 1-3，v4 涨到 5.5）
5. `reward/format_valid_rate` - 格式合法率（前 10-20 步应该快速上升）

---

## 本地单卡训练（快速验证）

如果想在本地单卡 GPU 上快速验证（不提交星云）：

```bash
# 1. 安装依赖（如果还没装）
bash exp/grpo/start/setup.sh

# 2. 运行训练
bash exp/grpo/start/train_local.sh
```

**本地 vs 星云的区别**：

| 配置 | 本地单卡 | 星云 4 卡 |
|------|---------|----------|
| Batch size | `train=8, mini=2` | `train=32, mini=8` |
| Rollout | `n=4, val_n=4` | `n=8, val_n=16` |
| Steps | 500 | 500 |
| GPU 内存 | `gpu_memory_utilization=0.4` | `0.6` |
| 数据路径 | `exp/grpo/data/ranking/` | OSS `/data/oss_bucket_0/ad/guoshauile.gsl/data/minirank/` |
| 模型路径 | `/home/guoshuaile.gsl/models/qwen3-8b` | OSS `/data/oss_bucket_0/ad/guoshauile.gsl/model/base/qwen3-8b` |

---

## 文件结构

```
exp/grpo/b/                     # ★ uv 环境版所有文件集中在此目录 ★
├── NEBULA_TRAINING.md          # 本文件（训练指南）
├── QUICK_REFERENCE.md          # 快速参考
├── grpo_ranking_gsl_v7_uv.sh   # 星云训练脚本（uv 环境版）
└── submit_nebula_uv.sh         # 星云提交脚本（无密钥，从环境变量读取）

exp/grpo/
├── start/
│   ├── train_local.sh          # 本地单卡训练脚本
│   ├── submit_nebula.sh        # 星云提交脚本（默认镜像版，含密钥，已 gitignore）
│   └── setup.sh                # 本地环境安装脚本
├── reward_fn.py                # v7 reward 实现（单条路径）
├── ranking_batch.py            # v7 reward 实现（batch 路径，训练实际走）
└── data/
    └── ranking/                # 本地数据（已 gitignore）

nebula_scripts/grpo/
└── grpo_ranking_gsl_v4.sh      # 星云训练脚本（默认镜像 + pip 版）
```

---

## 关键改动文件

### 1. Reward 函数

**`exp/grpo/reward_fn.py`** - 单条路径
```python
def _rule_based_score(...):
    # Top-3 Precision: 只看前 3 个位置
    # 正样本在 Top-3: pos0=1.0, pos1=0.5, pos2=0.33
    # 正样本不在 Top-3: 0（业务不曝光，无所谓）
    # 负样本在 Top-3: 加权惩罚
    # 负样本不在 Top-3: 0（业务不曝光，无所谓）
    # 归一化到 [0, 1]

def compute_reward(...):
    if not fc["valid"]:
        return {"score": 0.0, ...}  # DAPO filter
    
    top3_score = rule_result["score"]  # ∈ [0, 1]
    
    if has_judge:
        judge_norm = (judge_score + 1.0) / 2.0  # [-1,1] → [0,1]
        final_score = 0.7 * top3_score + 0.3 * judge_norm
    else:
        final_score = top3_score
```

**`exp/grpo/ranking_batch.py`** - Batch 路径（训练实际走这个）
- 同步了 v7 公式
- 使用 `_rule_based_score` 计算 Top-3 Precision

### 2. 训练脚本

**`exp/grpo/start/train_local.sh`** - 本地训练
```bash
python -m verl.trainer.main_ppo \
    --config-name baseline_grpo \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    algorithm.norm_adv_by_std_in_grpo=True \
    # ... 其他参数
```

**`nebula_scripts/grpo/grpo_ranking_gsl_v4.sh`** - 星云训练
- 同步了所有 v7 改动
- 硬编码了所有超参（不依赖环境变量）

### 3. 提交脚本

**`exp/grpo/start/submit_nebula.sh`** - 星云提交
- 包含密钥（OSS、OpenLM Token、SwanLab）
- 已加入 `.gitignore`，不会被提交
- 调用 `nebula_scripts/submit_job.sh`

---

## 验证清单

### 本地验证（单卡，快速）

```bash
# 1. 运行 10 步验证环境
TOTAL_TRAINING_STEPS=10 bash exp/grpo/start/train_local.sh

# 2. 检查日志中的关键指标
grep "actor/kl_loss" outputs/grpo_ranking_local/swanlab_logs/latest/files/config.yaml
# 应该看到 use_kl_loss: true

grep "score/mean" outputs/grpo_ranking_local/swanlab_logs/latest/files/config.yaml
# 应该看到 score 逐步上升

# 3. 检查 KL 散度
# rollout_corr/kl 应该 < 0.1（v4 爆炸到 0.7）
```

### 星云验证（4 卡，正式训练）

```bash
# 1. 提交任务
bash exp/grpo/start/submit_nebula.sh

# 2. 打开 LogView，检查 worker-0 日志
# - 前 50 步：format_valid_rate 应该快速上升到 >80%
# - 50-200 步：reward/rule_score/mean 应该逐步上升
# - 全程：rollout_corr/kl < 0.1, actor/entropy 在 1-3

# 3. 打开 SwanLab 查看完整曲线
# https://swanlab.cn
# Project: GRPO-Ranking
```

---

## 故障排查

### 问题 1: SwanLab 没有记录

**原因**：API key 错误
**解决**：检查 `submit_nebula.sh` 中的 `SWANLAB_API_KEY` 是否为 `o4MGQAOSX8rGztH69Jj5P`

### 问题 2: KL 散度爆炸（> 0.1）

**原因**：`use_kl_loss` 未开启
**解决**：检查训练日志中的配置，确认 `use_kl_loss: true`

### 问题 3: 模型输出乱码

**原因**：策略漂移，entropy 失控
**解决**：v7 已通过 KL 正则化 + DAPO filter 修复，如果仍然出现，尝试：
- 降低 `LR`（如 `5e-6`）
- 增加 `kl_loss_coef`（如 `0.02`）

### 问题 4: 格式合法率一直很低（< 50%）

**原因**：DAPO filter 的信号可能不够清晰
**解决**：考虑增加 SFT 预训练阶段，先教会模型输出合法 JSON

---

## 预期效果

### v4（旧版，84 步后崩溃）

```
Step 1:   score=-1.52, format_penalty=-1.40 (92%), KL=0.003
Step 50:  score=-0.58, format_penalty=-0.37 (64%), KL=0.007
Step 84:  score=-0.42, format_penalty=-0.31 (74%), KL=0.7 ❌ 爆炸
         输出变成多语言乱码
```

### v7（预期）

```
Step 1:   score=0.0, format_valid_rate=30%, KL=0.003
Step 10:  score=0.2, format_valid_rate=80%, KL=0.005
Step 50:  score=0.5, format_valid_rate=95%, KL=0.02
Step 200: score=0.7, format_valid_rate=98%, KL=0.05
Step 500: score=0.8, format_valid_rate=99%, KL=0.08 ✅ 稳定
```

**关键改进**：
1. `reward/rule_score/mean`（Top-3 Precision）应该逐步上升（v4 的纯排序质量 84 步没提升）
2. `rollout_corr/kl` 保持在 0.1 以下（v4 爆炸到 0.7）
3. `actor/entropy` 稳定在 1-3（v4 涨到 5.5）

---

## 参考文档

- [v7 Reward 设计说明](../../docs/v7_reward_redesign.md)
- [DAPO 论文](https://arxiv.org/abs/2501.12948) - Decoupled Clip and Dynamic Overlong Penalty
- [DeepSeek-R1 论文](https://arxiv.org/abs/2501.12948) - Incentivizing Reasoning Capability in LLMs
- [PRM 论文](https://arxiv.org/abs/2310.10080) - Process Reward Models

---

**最后更新**：2026-06-23  
**维护者**：guoshuaile.gsl
