#!/usr/bin/env bash
# =============================================================================
# GRPO Ranking 训练脚本 — guoshauile.gsl OSS 路径
#
# OSS 目录结构:
#   oss://lazada-ai-model/ad/guoshauile.gsl/
#     ├── data/          训练/测试数据
#     ├── model/         基底模型 & checkpoint
#     ├── log/           训练日志
#     └── result/        最终模型输出
# =============================================================================
set +xo pipefail

OSS_ROOT="/data/oss_bucket_0/ad/guoshauile.gsl"

# ── 从环境变量读取超参 ────────────────────────────────────────────────
: "${DATASET:?DATASET is not set}"
: "${LR:?LR is not set}"
: "${MINI_BATCH_SIZE:?MINI_BATCH_SIZE is not set}"
: "${TRAIN_BATCH_SIZE:?TRAIN_BATCH_SIZE is not set}"
: "${ROLLOUT_N:?ROLLOUT_N is not set}"
: "${MODEL_PATH:?MODEL_PATH is not set}"
KL_COEF="${KL_COEF:-0.05}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-500}"
N_GPUS="${N_GPUS:-4}"
VAL_N="${VAL_N:-16}"

# ── 路径 ────────────────────────────────────────────────────────────────
train_data_path="${OSS_ROOT}/data/${DATASET}/train.parquet"
val_data_path="${OSS_ROOT}/data/${DATASET}/test.parquet"
model_path="${MODEL_PATH}"
save_path="${OSS_ROOT}/result/${JOB_NAME:-grpo_ranking}"

# ── 环境 ────────────────────────────────────────────────────────────────
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# ── LLM Judge 配置 (qwen-plus via DashScope) ────────────────────────────
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-93bf8a433943448bad6611ca5532a113}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
export OPENAI_MODEL="${OPENAI_MODEL:-qwen3.6-max-preview}"

unset VLLM_ATTENTION_BACKEND
export VLLM_USE_V1=1
export VLLM_LOGGING_LEVEL=WARN
export RAY_memory_monitor_refresh_ms=0
export WANDB_MODE=offline
export WANDB_ENTITY=oh-my-team
export SWANLAB_MODE=cloud
export SWANLAB_API_KEY="${SWANLAB_API_KEY:-o4MGQAOSX8rGztH69Jj5P}"
export SWANLAB_LOG_DIR="${OSS_ROOT}/log/swanlab_logs"
export TORCH_WARN_ACCUMULATE_GRAD_STREAM=0

pip install -e . --no-deps --no-build-isolation --quiet 2>/dev/null || true

mkdir -p "${SWANLAB_LOG_DIR}" 2>/dev/null || true

ray stop --force 2>/dev/null || true
rm -rf /tmp/ray 2>/dev/null || true
sleep 3

python -m verl.trainer.main_ppo \
    --config-name baseline_grpo \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.train_files="${train_data_path}" \
    data.val_files="${val_data_path}" \
    reward_model.reward_manager=batch \
    custom_reward_function.path="$(pwd)/verl/utils/reward_score/feedback/__init__.py" \
    custom_reward_function.name=compute_score_batch \
    actor_rollout_ref.model.path="${model_path}" \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.val_kwargs.n=${VAL_N} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.kl_ctrl.kl_coef=${KL_COEF} \
    trainer.total_epochs=30 \
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS} \
    trainer.save_freq=-1 \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.val_before_train=False \
    trainer.default_local_dir="${save_path}" \
    trainer.project_name="${PROJECT_NAME:-GRPO-Ranking}" \
    trainer.experiment_name="${JOB_NAME:-grpo_ranking}" \
    trainer.group_name="GRPO-ranking" \
    "trainer.logger=[console]"
