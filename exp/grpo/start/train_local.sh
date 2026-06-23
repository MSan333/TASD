#!/usr/bin/env bash
# =============================================================================
# GRPO Ranking 本地单卡训练脚本（v7 — DAPO Filter + Top-3 Precision）
#
# 基于星云 v4 脚本改造，适配本地单 GPU 环境：
#   - 使用 uv 管理的 .venv 虚拟环境
#   - 本地数据路径 (exp/grpo/data/ranking/)
#   - 本地模型路径
#   - 单卡超参自动适配
#
# v7 修复（基于 DAPO/DeepSeek-R1/PRM 等论文 + v4 训练诊断）:
#   1. use_kl_loss=True — v4 的 use_kl_loss=False 导致 KL 正则化完全未开启
#   2. norm_adv_by_std_in_grpo=True — GRPO 标准做法
#   3. 去掉 rollout_is — 策略漂移后 IS ratio 从 1.0→0.2
#   4. reward: DAPO filter — 格式无效样本 score=0, GRPO 自动产生负 advantage
#   5. reward: Top-3 Precision — 只看前 3 个位置, 归一化到 [0, 1]
#   6. reward: 0.7 × top3 + 0.3 × judge_norm (judge 归一化到 [0, 1])
#
# 前置条件：
#   bash exp/grpo/start/setup.sh   # 一键安装 uv 环境
#
# 使用方式：
#   bash exp/grpo/start/train_local.sh
#
#   # 覆盖超参
#   LR=5e-6 TOTAL_TRAINING_STEPS=100 bash exp/grpo/start/train_local.sh
#
#   # 指定 GPU
#   CUDA_VISIBLE_DEVICES=0 bash exp/grpo/start/train_local.sh
# =============================================================================
set -euo pipefail

# ── 项目根目录 ──────────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${PROJECT_ROOT}"

# ── 激活 uv 虚拟环境 ───────────────────────────────────────────────────
if [ -d ".venv/bin" ]; then
    export PATH="${PROJECT_ROOT}/.venv/bin:${PATH}"
    echo "✅ 使用 uv 虚拟环境: .venv/"
elif [ -d "/opt/conda/envs/python3.10.13/bin" ]; then
    export PATH="/opt/conda/envs/python3.10.13/bin:${PATH}"
    echo "✅ 使用 conda 环境: python3.10.13"
else
    echo "❌ 未找到虚拟环境，请先运行: bash exp/grpo/start/setup.sh"
    exit 1
fi

# 验证关键依赖
python -c "import torch, ray, vllm, swanlab, verl" 2>/dev/null || {
    echo "❌ 缺少关键依赖，请先运行: bash exp/grpo/start/setup.sh"
    exit 1
}

# ── GPU 配置（默认单卡）──────────────────────────────────────────────────
N_GPUS="${N_GPUS:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

# ── 超参配置（v7 DAPO + Top-3，单卡适配）────────────────────────────────
#
# v7 修复 (基于 DAPO/DeepSeek-R1/PRM 等论文 + v4 训练诊断):
#   1. use_kl_loss=True + kl_loss_coef=0.01 — v4 的 use_kl_loss=False 导致
#      KL 正则化完全未开启 (actor/kl_loss=0.0 全程)，KL 从 0.003 爆炸到 0.7
#   2. norm_adv_by_std_in_grpo=True — GRPO 标准做法，稳定 advantage 信号
#   3. 去掉 rollout_is — 策略漂移后 IS ratio 从 1.0→0.2，修正变成噪声
#   4. reward: DAPO filter (invalid=0) + Top-3 Precision ([0,1])
#   5. reward: 0.7 × top3 + 0.3 × judge_norm
#
DATASET="${DATASET:-ranking}"
LR="${LR:-1e-5}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.003}"
ENTROPY_COEFF="${ENTROPY_COEFF:-0.02}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-500}"

# 单卡 H20 144GB 配置
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
ROLLOUT_N="${ROLLOUT_N:-8}"
VAL_N="${VAL_N:-4}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.5}"

# 序列长度（与 v4 一致）
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

# 生成参数（提高 top_p 增加多样性，解决 entropy 过低问题）
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.9}"
TOP_K="${TOP_K:--1}"

# 模型路径
MODEL_PATH="${MODEL_PATH:-/home/guoshuaile.gsl/models/qwen3-8b}"

# ── 路径 ────────────────────────────────────────────────────────────────
train_data_path="${PROJECT_ROOT}/exp/grpo/data/${DATASET}/train.parquet"
val_data_path="${PROJECT_ROOT}/exp/grpo/data/${DATASET}/test.parquet"
save_path="${PROJECT_ROOT}/outputs/grpo_ranking_local"

# 验证数据和模型
for f in "${train_data_path}" "${val_data_path}"; do
    if [ ! -f "${f}" ]; then
        echo "❌ 数据文件不存在: ${f}"
        exit 1
    fi
done
if [ ! -d "${MODEL_PATH}" ]; then
    echo "❌ 模型路径不存在: ${MODEL_PATH}"
    exit 1
fi

# ── 环境变量（对齐 v4）──────────────────────────────────────────────────
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# LLM Judge (DashScope)
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-93bf8a433943448bad6611ca5532a113}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
export OPENAI_MODEL="${OPENAI_MODEL:-qwen3.6-max-preview}"

# vLLM
unset VLLM_ATTENTION_BACKEND
export VLLM_USE_V1=1
export VLLM_LOGGING_LEVEL=WARN

# Ray
export RAY_memory_monitor_refresh_ms=0

# Wandb 关闭（使用 SwanLab 替代）
export WANDB_MODE=offline

# SwanLab
export SWANLAB_MODE="${SWANLAB_MODE:-cloud}"
export SWANLAB_API_KEY="${SWANLAB_API_KEY:-o4MGQAOSX8rGztH69Jj5P}"
export SWANLAB_LOG_DIR="${save_path}/swanlab_logs"

# PyTorch
export TORCH_WARN_ACCUMULATE_GRAD_STREAM=0

# ── 安装项目（开发模式）──────────────────────────────────────────────────
pip install -e . --no-deps --no-build-isolation --quiet 2>/dev/null || true

# ── 创建输出目录 ────────────────────────────────────────────────────────
mkdir -p "${save_path}" "${SWANLAB_LOG_DIR}" 2>/dev/null || true

# ── 清理 Ray 残留 ──────────────────────────────────────────────────────
ray stop --force 2>/dev/null || true
rm -rf /tmp/ray 2>/dev/null || true
sleep 3

# ── 实验名称 ────────────────────────────────────────────────────────────
JOB_NAME="${JOB_NAME:-grpo_ranking_local_${N_GPUS}gpu_$(date +%Y%m%d_%H%M%S)}"
PROJECT_NAME="${PROJECT_NAME:-GRPO-Ranking}"

# ── 打印配置 ────────────────────────────────────────────────────────────
echo "============================================================"
echo "  GRPO Ranking 本地训练（v7 DAPO+Top3）"
echo "  实验名称 : ${JOB_NAME}"
echo "  模型     : ${MODEL_PATH}"
echo "  训练数据 : ${train_data_path}"
echo "  验证数据 : ${val_data_path}"
echo "  GPU      : ${N_GPUS} 卡 (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
echo "  Batch    : train=${TRAIN_BATCH_SIZE}, mini=${MINI_BATCH_SIZE}"
echo "  Rollout  : n=${ROLLOUT_N}, val_n=${VAL_N}"
echo "  序列长度 : prompt=${MAX_PROMPT_LENGTH}, response=${MAX_RESPONSE_LENGTH}, model=${MAX_MODEL_LEN}"
echo "  生成参数 : temp=${TEMPERATURE}, top_p=${TOP_P}, top_k=${TOP_K}"
echo "  LR       : ${LR}"
echo "  KL Loss  : use_kl_loss=True, kl_loss_coef=${KL_LOSS_COEF}"
echo "  Entropy  : ${ENTROPY_COEFF}"
echo "  Steps    : ${TOTAL_TRAINING_STEPS}"
echo "  SwanLab  : ${SWANLAB_MODE}"
echo "  输出路径 : ${save_path}"
echo "============================================================"

# ── 启动训练（v7 DAPO Filter + Top-3 Precision）────────────────────────
# 关键改动 vs v4:
#   - use_kl_loss=True + kl_loss_coef=0.01 (v4 的 KL 完全未开启!)
#   - norm_adv_by_std_in_grpo=True (GRPO 标准做法)
#   - 去掉 rollout_correction.rollout_is (策略漂移后 IS 变成噪声)
#   - reward: DAPO filter (invalid=0) + Top-3 Precision ([0,1]) + judge_norm
python -m verl.trainer.main_ppo \
    --config-name baseline_grpo \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.train_files="${train_data_path}" \
    data.val_files="${val_data_path}" \
    data.val_max_samples=100 \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    max_model_len=${MAX_MODEL_LEN} \
    reward_model.reward_manager=batch \
    custom_reward_function.path="${PROJECT_ROOT}/verl/utils/reward_score/feedback/__init__.py" \
    custom_reward_function.name=compute_score_batch \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.val_kwargs.n=${VAL_N} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${TEMPERATURE} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${TOP_P} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${TOP_K} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL} \
    actor_rollout_ref.rollout.temperature=${TEMPERATURE} \
    actor_rollout_ref.rollout.top_p=${TOP_P} \
    actor_rollout_ref.rollout.top_k=${TOP_K} \
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF} \
    algorithm.norm_adv_by_std_in_grpo=True \
    trainer.total_epochs=30 \
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS} \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.save_best_metric=val/test_score/mean \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.val_before_train=False \
    trainer.default_local_dir="${save_path}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${JOB_NAME}" \
    trainer.group_name="GRPO-ranking-local" \
    "trainer.logger=[console,swanlab]"
