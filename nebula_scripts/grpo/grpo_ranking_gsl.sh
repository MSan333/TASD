#!/usr/bin/env bash
# =============================================================================
# GRPO Ranking 训练脚本 — guoshauile.gsl OSS 路径
#
# ★ 所有配置全部硬编码在脚本内，不依赖任何外部环境变量传递 ★
# ★ 修改超参直接改下方数值即可 ★
#
# OSS 目录结构:
#   oss://lazada-ai-model/ad/guoshauile.gsl/
#     ├── data/          训练/测试数据
#     ├── model/         基底模型 & checkpoint
#     ├── log/           训练日志
#     └── result/        最终模型输出
# =============================================================================
set +xo pipefail

# ── 训练超参（硬编码，改这里即可）──────────────────────────────────────
DATASET="ranking"
LR="1e-5"
MINI_BATCH_SIZE="8"
TRAIN_BATCH_SIZE="32"
ROLLOUT_N="8"
KL_COEF="0.05"
TOTAL_TRAINING_STEPS="500"
N_GPUS="4"
VAL_N="16"
PROJECT_NAME="GRPO-Ranking"

# ── 路径（硬编码）──────────────────────────────────────────────────────
OSS_ROOT="/data/oss_bucket_0/ad/guoshauile.gsl"
MODEL_PATH="/data/oss_bucket_0/ad/guoshauile.gsl/model/base/qwen3-8b"

train_data_path="${OSS_ROOT}/data/${DATASET}/train.parquet"
val_data_path="${OSS_ROOT}/data/${DATASET}/test.parquet"
model_path="${MODEL_PATH}"
save_path="${OSS_ROOT}/result/${JOB_NAME:-grpo_ranking}"

# ── 环境变量（硬编码）─────────────────────────────────────────────────
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# LLM Judge (DashScope)
# ★ 如需使用 LLM Judge，请设置 OPENAI_API_KEY 环境变量 ★
# export OPENAI_API_KEY="你的dashscope_api_key"
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export OPENAI_MODEL="qwen3.6-max-preview"

# vLLM
unset VLLM_ATTENTION_BACKEND
export VLLM_USE_V1=1
export VLLM_LOGGING_LEVEL=WARN

# Ray
export RAY_memory_monitor_refresh_ms=0

# W&B
export WANDB_MODE=offline
export WANDB_ENTITY=oh-my-team

# SwanLab（API Key 可直接明文，无安全风险）
export SWANLAB_MODE=cloud
export SWANLAB_API_KEY="o4MGQAOSX8rGztH69Jj5P"
export SWANLAB_LOG_DIR="${OSS_ROOT}/log/swanlab_logs"

# PyTorch
export TORCH_WARN_ACCUMULATE_GRAD_STREAM=0

# ── 安装依赖 & 项目 ──────────────────────────────────────────────────
echo "=== 检查当前环境 ==="
python -c "import transformers; print(f'  transformers={transformers.__version__}')" 2>&1 || echo "  transformers 未安装"
python -c "import vllm; print(f'  vllm={vllm.__version__}')" 2>&1 || echo "  vllm 未安装"

echo "=== 安装 TASD 项目（不拉依赖） ==="
pip install -e . --no-deps --no-build-isolation 2>&1

# 使用 transformers 4.53.2（支持 qwen3 + layer_type_validation + 本地路径）
echo "=== 安装/升级关键依赖（锁定版本） ==="
pip install \
    "transformers==4.53.2" \
    "huggingface-hub==0.30.2" \
    "numpy<2" \
    "tensordict>=0.8.0,<=0.10.0,!=0.9.0" \
    "hydra-core" \
    "omegaconf" \
    "codetiming" \
    "dill" \
    "pybind11" \
    "pylatexenc" \
    "torchdata" \
    "peft" \
    "swanlab" \
    "openlm-hub" \
    "liger-kernel" \
    "openai" \
    "accelerate" \
    "datasets" \
    "ray[default]>=2.41.0" \
    "pyarrow>=19.0.0" \
    "wandb" \
    "tensorboard" \
    "math-verify[antlr4_9_3]" \
    "latex2sympy2_extended" \
    "latex2sympy2" \
    "word2number" \
    2>&1

# 验证关键依赖（包括 verl 实际用到的 import）
echo "=== 验证关键依赖 ==="
python -c "
from transformers import AutoModelForImageTextToText, AutoModelForVision2Seq
print('  ✅ transformers AutoModelForImageTextToText + AutoModelForVision2Seq')
import transformers; print(f'  transformers={transformers.__version__}')
import numpy; print(f'  numpy={numpy.__version__}')
import tensordict; print(f'  tensordict={tensordict.__version__}')
import omegaconf; print(f'  omegaconf={omegaconf.__version__}')
import codetiming; print('  codetiming=OK')
import verl; print('  verl=OK')
print('✅ 所有关键依赖导入成功')
" || { echo "❌ 依赖验证失败，退出"; exit 1; }

# 验证模型路径是否存在
echo "=== 验证模型路径 ==="
if [ -d "$MODEL_PATH" ]; then
    echo "  ✅ 模型路径存在: $MODEL_PATH"
    ls -lh "$MODEL_PATH/" | head -10
else
    echo "  ❌ 模型路径不存在: $MODEL_PATH"
    exit 1
fi

mkdir -p "${SWANLAB_LOG_DIR}" 2>/dev/null || true

ray stop --force 2>/dev/null || true
rm -rf /tmp/ray 2>/dev/null || true
sleep 3

# ── 打印配置（方便排查）──────────────────────────────────────────────
echo "============================================================"
echo "GRPO Ranking 训练配置"
echo "  DATASET             = ${DATASET}"
echo "  LR                  = ${LR}"
echo "  TRAIN_BATCH_SIZE    = ${TRAIN_BATCH_SIZE}"
echo "  MINI_BATCH_SIZE     = ${MINI_BATCH_SIZE}"
echo "  ROLLOUT_N           = ${ROLLOUT_N}"
echo "  KL_COEF             = ${KL_COEF}"
echo "  TOTAL_TRAINING_STEPS= ${TOTAL_TRAINING_STEPS}"
echo "  VAL_N               = ${VAL_N}"
echo "  MODEL_PATH          = ${model_path}"
echo "  train_data          = ${train_data_path}"
echo "  val_data            = ${val_data_path}"
echo "  JOB_NAME            = ${JOB_NAME:-grpo_ranking}"
echo "============================================================"

# ── 启动训练 ──────────────────────────────────────────────────────────
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
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${JOB_NAME:-grpo_ranking}" \
    trainer.group_name="GRPO-ranking" \
    "trainer.logger=[console,swanlab]"
