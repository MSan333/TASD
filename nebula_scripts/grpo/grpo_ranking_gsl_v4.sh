#!/usr/bin/env bash
# =============================================================================
# GRPO Ranking v5 训练脚本 — 统一公式版
#
# 奖励函数（统一公式）:
#   final_score = α × rule_score + (1-α) × judge_score + format_penalty
#
#   - 有 LLM Judge: α=0.5 → final = 0.5×rule + 0.5×judge + format_penalty
#   - 无 LLM Judge: α=1.0 → final = rule_score + format_penalty
#   - 格式不合法:   同样算 rule_score（用已解析部分），加更重的 format_penalty
#
# 格式惩罚（较轻，辅助信号）:
#   - 完全无效（0张卡）: -0.3
#   - 部分有效（<3张卡）: -0.15
#   - Markdown 包裹:     -0.1
#   - 幻觉卡片:          -0.15/个, max -0.25
#   - 遗漏卡片:          -0.025/个, max -0.15
#   - 总惩罚上限:        -0.4
#
# ★ 所有配置全部硬编码在脚本内，不依赖任何外部环境变量传递 ★
# ★ 修改超参直接改下方数值即可 ★
# =============================================================================
set +xo pipefail

# ── 训练超参（硬编码，改这里即可）──────────────────────────────────────
DATASET="minirank"
LR="1e-5"
MINI_BATCH_SIZE="8"
TRAIN_BATCH_SIZE="32"
ROLLOUT_N="8"
KL_COEF="0.05"
ENTROPY_COEFF="0.01"
TOTAL_TRAINING_STEPS="500"
N_GPUS="4"
VAL_N="16"
PROJECT_NAME="GRPO-Ranking"

# ── 路径（硬编码）──────────────────────────────────────────────────────
OSS_ROOT="/data/oss_bucket_0/ad/guoshauile.gsl"
MODEL_PATH="/data/oss_bucket_0/ad/guoshauile.gsl/model/base/qwen3-8b"

train_data_path="${OSS_ROOT}/data/${DATASET}/train.parquet"
val_data_path="${OSS_ROOT}/data/${DATASET}/test_small.parquet"  # 100 prompts (原 1982)
model_path="${MODEL_PATH}"
save_path="${OSS_ROOT}/result/${JOB_NAME:-grpo_ranking}"

# ── 环境变量（硬编码）─────────────────────────────────────────────────
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# LLM Judge (DashScope)
export OPENAI_API_KEY="sk-93bf8a433943448bad6611ca5532a113"
export OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export JUDGE_MODEL="qwen3.6-max-preview"

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
echo "  ENTROPY_COEFF       = ${ENTROPY_COEFF}"
echo "  JUDGE_MODEL         = ${JUDGE_MODEL}"
echo "  TOTAL_TRAINING_STEPS= ${TOTAL_TRAINING_STEPS}"
echo "  VAL_N               = ${VAL_N}"
echo "  MODEL_PATH          = ${model_path}"
echo "  train_data          = ${train_data_path}"
echo "  val_data            = ${val_data_path}"
echo "  JOB_NAME            = ${JOB_NAME:-grpo_ranking}"
echo "============================================================"

# ── 数据集分析（训练前诊断）───────────────────────────────────────────
echo "=== 数据集分析 ==="
python3 - "${train_data_path}" "${val_data_path}" << 'DATAEOF'
import sys, json
import pandas as pd
from collections import Counter

train_path, val_path = sys.argv[1], sys.argv[2]

for split, path in [("train", train_path), ("val", val_path)]:
    try:
        df = pd.read_parquet(path)
        print(f"\n[{split}] 样本数: {len(df)}")

        if "prompt" in df.columns:
            n_unique = df["prompt"].nunique()
            print(f"  唯一 prompt 数: {n_unique}")
            print(f"  每 prompt 平均样本: {len(df)/n_unique:.1f}")

        # ground_truth 在 reward_model 列内
        if "reward_model" in df.columns:
            gts = df["reward_model"].apply(
                lambda x: json.loads(x["ground_truth"]) if isinstance(x, dict) else json.loads(x)
            )

            pos_lens = gts.apply(lambda x: len(x.get("positive_actions", [])))
            pos_empty = (pos_lens == 0).mean()
            print(f"  positive_actions 空比例: {pos_empty:.1%}")
            print(f"  positive_actions 长度分布: {dict(Counter(pos_lens))}")

            neg_lens = gts.apply(lambda x: len(x.get("negative_keys", [])))
            neg_empty = (neg_lens == 0).mean()
            print(f"  negative_keys 空比例: {neg_empty:.1%}")

            pools = gts.apply(lambda x: tuple(sorted(x.get("card_pool", []))))
            print(f"  唯一 card_pool 数: {pools.nunique()}")
    except Exception as e:
        print(f"[{split}] 分析失败: {e}")
DATAEOF

# ── 启动训练 ──────────────────────────────────────────────────────────
python -m verl.trainer.main_ppo \
    --config-name baseline_grpo \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.train_files="${train_data_path}" \
    data.val_files="${val_data_path}" \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    max_model_len=8192 \
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
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF} \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.kl_ctrl.kl_coef=${KL_COEF} \
    trainer.total_epochs=30 \
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS} \
    trainer.save_freq=10 \
    trainer.save_best_metric=val/test_score/mean \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.val_before_train=False \
    trainer.default_local_dir="${save_path}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${JOB_NAME:-grpo_ranking}" \
    trainer.group_name="GRPO-ranking" \
    "trainer.logger=[console,swanlab]"
