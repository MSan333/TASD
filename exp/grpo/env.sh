#!/usr/bin/env bash
# =============================================================================
# GRPO Ranking 训练 — 环境变量配置
#
# 使用方式:
#
#   ┌─────────────────────────────────────────────────────────────────┐
#   │ 场景 A: 星云提交 (4 卡 H20)                                     │
#   │                                                                 │
#   │   source exp/grpo/env.sh                                        │
#   │   bash nebula_scripts/submit_job.sh \                           │
#   │       nebula_scripts/grpo/grpo_ranking_gsl.sh \                 │
#   │       1 \                                                       │
#   │       lazada_llm_ad_h20                                         │
#   │                                                                 │
#   │ 场景 B: 本地训练 (自动适配 GPU 数量)                              │
#   │                                                                 │
#   │   conda activate python3.10                                     │
#   │   source exp/grpo/env.sh                                        │
#   │   bash run_local_grpo_ranking.sh                                │
#   │                                                                 │
#   │ 场景 C: 覆盖超参                                                 │
#   │                                                                 │
#   │   source exp/grpo/env.sh                                        │
#   │   LR=5e-6 TRAIN_BATCH_SIZE=16 bash run_local_grpo_ranking.sh   │
#   └─────────────────────────────────────────────────────────────────┘
#
# 修改超参:
#   直接在下方修改对应变量的值, 或在命令行用 VAR=value 临时覆盖。
#   例如只改学习率: LR=5e-6 bash run_local_grpo_ranking.sh
#
# =============================================================================

# ── 训练超参 ────────────────────────────────────────────────────────────
# 这些是 4 卡星云环境的标准配置。
# 本地训练时, run_local_grpo_ranking.sh 会根据 GPU 数量自动降档:
#   4卡: batch=32, mini=8, rollout=8, val_n=16 (与星云一致)
#   2卡: batch=16, mini=4, rollout=4, val_n=8
#   1卡: batch=8,  mini=2, rollout=4, val_n=4
export DATASET="ranking"
export LR="1e-5"
export MINI_BATCH_SIZE="8"
export TRAIN_BATCH_SIZE="32"
export ROLLOUT_N="8"
export KL_COEF="0.05"
export TOTAL_TRAINING_STEPS="500"
export VAL_N="16"

# ── 模型路径 ────────────────────────────────────────────────────────────
# 星云环境使用 OSS 挂载路径:
export MODEL_PATH="/data/oss_bucket_0/ad/guoshauile.gsl/model/qwen3-8b"
# 如果是本地训练, run_local_grpo_ranking.sh 的默认值是:
#   /home/guoshuaile.gsl/models/qwen3-8b
# 可通过环境变量覆盖: MODEL_PATH=/your/local/model bash run_local_grpo_ranking.sh

# ── 星云/Nebula 账号 ───────────────────────────────────────────────────
# 提交星云任务前, 必须设置以下 3 个环境变量。
# 建议写入 ~/.bashrc 避免每次手动设置:
#   export OPENLM_TOKEN="your_openlm_token"
#   export OSS_ACCESS_ID="your_oss_access_id"
#   export OSS_ACCESS_KEY="your_oss_access_key"

# ── LLM Judge (DashScope) ─────────────────────────────────────────────
# Reward 函数中的 Knowledge-Grounded Judge 使用 DashScope API 打分。
# ★ 请设置环境变量（到 https://dashscope.console.aliyun.com 获取）★
# export OPENAI_API_KEY="你的dashscope_api_key"
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
export OPENAI_MODEL="${OPENAI_MODEL:-qwen3.6-max-preview}"

# ── SwanLab 日志追踪 ──────────────────────────────────────────────────
# SwanLab Cloud 模式: 训练指标自动上传到 https://swanlab.cn
# API Key 可直接明文，无安全风险。如需更换到 https://swanlab.cn/settings 获取。
export SWANLAB_API_KEY="o4MGQAOSX8rGztH69Jj5P"
export SWANLAB_MODE="cloud"

# ── 实验名称 ───────────────────────────────────────────────────────────
export PROJECT_NAME="GRPO-Ranking"
export JOB_NAME="${JOB_NAME:-grpo_ranking_$(date +%Y%m%d_%H%M%S)}"

echo "✅ GRPO Ranking 环境变量已加载"
echo "   DATASET=${DATASET}, LR=${LR}, BATCH=${TRAIN_BATCH_SIZE}, ROLLOUT_N=${ROLLOUT_N}"
echo "   MODEL_PATH=${MODEL_PATH}"
echo "   JOB_NAME=${JOB_NAME}"
