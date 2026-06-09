#!/usr/bin/env bash
# =============================================================================
# 本地 GRPO Ranking 训练脚本
#
# 基于 nebula_scripts/grpo/grpo_ranking_gsl.sh 改造，适配:
#   - 自动检测 GPU 数量 (1卡/4卡/8卡均可)
#   - 本地数据路径 (exp/grpo/data/ranking/)
#   - SwanLab cloud 模式日志上报
#   - 使用 conda python3.10 环境 (含 torch/ray/vllm/swanlab)
#
# 使用方式:
#   # 方式1: 直接运行 (自动检测GPU, 使用默认超参)
#   bash run_local_grpo_ranking.sh
#
#   # 方式2: 配合 env.sh (可覆盖超参)
#   source exp/grpo/env.sh
#   bash run_local_grpo_ranking.sh
#
#   # 方式3: 指定GPU数量
#   N_GPUS=4 bash run_local_grpo_ranking.sh
#
#   # 方式4: 指定可见GPU
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_local_grpo_ranking.sh
# =============================================================================
set -euo pipefail

# ── 项目根目录 ──────────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

# ── 使用 conda python3.10 环境 ─────────────────────────────────────────
CONDA_PYTHON="/opt/conda/envs/python3.10/bin/python"
if [ -x "${CONDA_PYTHON}" ]; then
    export PATH="/opt/conda/envs/python3.10/bin:${PATH}"
fi
# 验证关键依赖
python -c "import torch, ray, vllm, swanlab" 2>/dev/null || {
    echo "❌ 缺少关键依赖 (torch/ray/vllm/swanlab)"
    echo "   请确保 conda python3.10 环境已安装: conda activate python3.10"
    exit 1
}

# ── 自动检测 GPU 数量 ──────────────────────────────────────────────────
if [ -z "${N_GPUS:-}" ]; then
    N_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
fi

# ── 根据 GPU 数量自动调整超参 ──────────────────────────────────────────
DATASET="${DATASET:-ranking}"
LR="${LR:-1e-5}"
KL_COEF="${KL_COEF:-0.05}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-500}"
MODEL_PATH="${MODEL_PATH:-/home/guoshuaile.gsl/models/qwen3-8b}"

if [ "${N_GPUS}" -ge 4 ]; then
    # 4卡配置 (与星云一致)
    MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-8}"
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
    ROLLOUT_N="${ROLLOUT_N:-8}"
    VAL_N="${VAL_N:-16}"
    GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.5}"
elif [ "${N_GPUS}" -ge 2 ]; then
    # 2卡配置
    MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-4}"
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
    ROLLOUT_N="${ROLLOUT_N:-4}"
    VAL_N="${VAL_N:-8}"
    GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.5}"
else
    # 1卡配置
    MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-2}"
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
    ROLLOUT_N="${ROLLOUT_N:-4}"
    VAL_N="${VAL_N:-4}"
    GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.5}"
fi

# ── 数据路径 (本地) ─────────────────────────────────────────────────────
train_data_path="${PROJECT_ROOT}/exp/grpo/data/${DATASET}/train.parquet"
val_data_path="${PROJECT_ROOT}/exp/grpo/data/${DATASET}/test.parquet"
save_path="${PROJECT_ROOT}/outputs/grpo_ranking_local"

# ── 验证数据和模型存在 ──────────────────────────────────────────────────
for required_file in "${train_data_path}" "${val_data_path}"; do
    if [ ! -f "${required_file}" ]; then
        echo "❌ 数据文件不存在: ${required_file}"
        echo "   请先下载: python -m exp.grpo.download_data"
        exit 1
    fi
done
if [ ! -d "${MODEL_PATH}" ]; then
    echo "❌ 模型路径不存在: ${MODEL_PATH}"
    exit 1
fi

# ── 环境变量 ────────────────────────────────────────────────────────────
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# LLM Judge 配置 (qwen-plus via DashScope)
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-93bf8a433943448bad6611ca5532a113}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
export OPENAI_MODEL="${OPENAI_MODEL:-qwen3.6-max-preview}"

# vLLM 配置
unset VLLM_ATTENTION_BACKEND
export VLLM_USE_V1=1
export VLLM_LOGGING_LEVEL=WARN

# Ray 配置
export RAY_memory_monitor_refresh_ms=0

# Wandb 关闭 (使用 SwanLab 替代)
export WANDB_MODE=offline

# SwanLab 配置 — cloud 模式
export SWANLAB_MODE=cloud
export SWANLAB_API_KEY="${SWANLAB_API_KEY:-o4MGQAOSX8rGztH69Jj5P}"
export SWANLAB_LOG_DIR="${save_path}/swanlab_logs"

# 其他
export TORCH_WARN_ACCUMULATE_GRAD_STREAM=0

# ── 安装项目 (如未安装) ─────────────────────────────────────────────────
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

echo "============================================================"
echo "  本地 GRPO Ranking 训练"
echo "  实验名称 : ${JOB_NAME}"
echo "  模型     : ${MODEL_PATH}"
echo "  训练数据 : ${train_data_path}"
echo "  验证数据 : ${val_data_path}"
echo "  GPU 数量 : ${N_GPUS}"
echo "  Batch    : train=${TRAIN_BATCH_SIZE}, mini=${MINI_BATCH_SIZE}"
echo "  Rollout  : n=${ROLLOUT_N}, val_n=${VAL_N}"
echo "  LR       : ${LR}"
echo "  KL Coef  : ${KL_COEF}"
echo "  Steps    : ${TOTAL_TRAINING_STEPS}"
echo "  SwanLab  : ${SWANLAB_MODE}"
echo "  输出路径 : ${save_path}"
echo "============================================================"

# ── 启动训练 ────────────────────────────────────────────────────────────
python -m verl.trainer.main_ppo \
    --config-name baseline_grpo \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.train_files="${train_data_path}" \
    data.val_files="${val_data_path}" \
    reward_model.reward_manager=batch \
    custom_reward_function.path="${PROJECT_ROOT}/verl/utils/reward_score/feedback/__init__.py" \
    custom_reward_function.name=compute_score_batch \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.val_kwargs.n=${VAL_N} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL} \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.kl_ctrl.kl_coef=${KL_COEF} \
    trainer.total_epochs=30 \
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS} \
    trainer.save_freq=-1 \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.val_before_train=False \
    trainer.default_local_dir="${save_path}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${JOB_NAME}" \
    trainer.group_name="GRPO-ranking-local" \
    "trainer.logger=[console,swanlab]"
