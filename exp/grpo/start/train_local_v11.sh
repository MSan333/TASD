#!/usr/bin/env bash
# =============================================================================
# GRPO Ranking 本地训练脚本（v11 — Pairwise Preference Reward）
#
# v11 设计 (基于 v10 训练诊断 19 步):
#   v10 问题:
#     - judge 打分无区分度: avg_llm ∈ [-0.14, -0.74], 中位数 ~ -0.55
#       → judge_norm 集中在 [0.15, 0.25], GRPO 组内差异 ≈ 0
#       → advantage ≈ 0 → pg_loss ≈ 0 → 策略不更新
#     - entropy 坍缩: 0.465 → 0.074 (85% 坍缩)
#       → entropy_coeff=0 无探索正则, 模型输出趋于确定
#     - 模型只学会 "输出合法 JSON + 拿 clean bonus", 排序质量未改善
#   v11 改进:
#   1. Pairwise Preference Reward: judge 输出离散 preference {-1, 0, +1}
#      - 相对比较替代绝对打分, 强制离散输出, 天然有方差
#      - 组内 pairwise 比较: 每个 rollout vs 中位数锚点
#      - 解决 v10 Judge 无区分度问题
#   2. entropy_coeff: 0 → 0.02 (防止 entropy 坍缩)
#   3. rollout_n: 4 → 8 (更多 rollout, pairwise 比较更充分)
#
# 公式: score = base_score + α × judge_preference + format_penalty + clean_bonus
#   - base_score: rule_score (确定性, 无 LLM)
#   - judge_preference: pairwise 比较 {-1, 0, +1}
#   - α = 0.3: pairwise 权重
#   - format_penalty: markdown -0.1, extra_text -0.05
#   - clean_bonus: 纯 JSON +0.1
#   - invalid: progressive penalty (-0.3 ~ 0.15)
#
# 预期效果:
#   - judge_preference 分布均匀: {-1: ~33%, 0: ~33%, +1: ~33%}
#   - entropy 稳定在 [0.3, 0.5] (不坍缩)
#   - pg_loss 有非零梯度 (>0.01)
#   - base_score 逐步上升 (排序质量改善)
#
# 前置条件:
#   bash exp/grpo/start/setup.sh   # 一键安装 uv 环境
#
# 使用方式:
#   bash exp/grpo/start/train_local_v11.sh
#
#   # 覆盖超参
#   LR=5e-6 TOTAL_TRAINING_STEPS=100 bash exp/grpo/start/train_local_v11.sh
#
#   # 指定 GPU
#   CUDA_VISIBLE_DEVICES=0 bash exp/grpo/start/train_local_v11.sh
#
#   # 指定数据集 (ranking 或 minirank)
#   DATASET=minirank bash exp/grpo/start/train_local_v11.sh
# =============================================================================
set -euo pipefail

# ── 项目根目录 ──────────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${PROJECT_ROOT}"

# ── 清理 __pycache__ (防止 Ray worker 加载旧字节码) ─────────────────────
echo "🧹 清理 __pycache__ ..."
find "${PROJECT_ROOT}/exp/grpo" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find "${PROJECT_ROOT}/verl/utils/reward_score" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# ── 激活虚拟环境 ─────────────────────────────────────────────────────────
if [ -d ".venv/bin" ]; then
    export PATH="${PROJECT_ROOT}/.venv/bin:${PATH}"
    echo "✅ 使用 uv 虚拟环境: .venv/"
elif [ -d "/opt/conda/envs/python3.10.13/bin" ]; then
    export PATH="/opt/conda/envs/python3.10.13/bin:${PATH}"
    echo "✅ 使用 conda 环境: python3.10.13"
elif [ -d "/opt/conda/envs/python3.10/bin" ]; then
    export PATH="/opt/conda/envs/python3.10/bin:${PATH}"
    echo "✅ 使用 conda 环境: python3.10"
else
    echo "❌ 未找到虚拟环境，请先运行: bash exp/grpo/start/setup.sh"
    exit 1
fi

# 验证关键依赖
python -c "import torch, ray, vllm, swanlab, verl" 2>/dev/null || {
    echo "❌ 缺少关键依赖，请先运行: bash exp/grpo/start/setup.sh"
    exit 1
}

# ── GPU 配置（自动检测）──────────────────────────────────────────────────
if [ -z "${N_GPUS:-}" ]; then
    N_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
fi
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

# ── 数据集 ───────────────────────────────────────────────────────────────
DATASET="${DATASET:-minirank}"

# ── 路径配置: 优先 OSS 挂载, 无挂载则用本地 ──────────────────────────────
OSS_MOUNTED=false
OSS_ROOT=""

for candidate in "/data/oss_bucket_0/ad/guoshauile.gsl" "/mnt/oss/ad/guoshauile.gsl" "/mnt/oss"; do
    # 检查目录存在且可读（fuse 挂载可能目录存在但 I/O 报错）
    if [ -d "${candidate}" ] && ls "${candidate}" >/dev/null 2>&1; then
        OSS_ROOT="${candidate}"
        OSS_MOUNTED=true
        break
    fi
done

if [ "${OSS_MOUNTED}" = true ]; then
    echo "📦 使用 OSS 挂载路径: ${OSS_ROOT}"
    MODEL_PATH="${MODEL_PATH:-${OSS_ROOT}/model/base/qwen3-8b}"
    train_data_path="${train_data_path:-${OSS_ROOT}/data/${DATASET}/train.parquet}"
    val_data_path="${val_data_path:-${OSS_ROOT}/data/${DATASET}/test_small.parquet}"
    save_path="${save_path:-${OSS_ROOT}/result/grpo_ranking_v11}"
else
    echo "📁 OSS 未挂载, 使用本地路径"
    MODEL_PATH="${MODEL_PATH:-/home/guoshuaile.gsl/models/qwen3-8b}"
    train_data_path="${train_data_path:-${PROJECT_ROOT}/exp/grpo/data/${DATASET}/train.parquet}"
    val_data_path="${val_data_path:-${PROJECT_ROOT}/exp/grpo/data/${DATASET}/test_small.parquet}"
    save_path="${save_path:-${PROJECT_ROOT}/outputs/grpo_ranking_v11}"
fi

# ── 超参配置（v11: Pairwise Preference + entropy boost, 按 GPU 数量自动适配）──
LR="${LR:-1e-5}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.01}"
ENTROPY_COEFF="${ENTROPY_COEFF:-0.02}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-500}"

if [ "${N_GPUS}" -ge 4 ]; then
    MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-8}"
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
    ROLLOUT_N="${ROLLOUT_N:-8}"
    VAL_N="${VAL_N:-16}"
    GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.5}"
elif [ "${N_GPUS}" -ge 2 ]; then
    MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-4}"
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
    ROLLOUT_N="${ROLLOUT_N:-4}"
    VAL_N="${VAL_N:-8}"
    GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.5}"
else
    # 单卡配置 (低内存模式, 避免 OOM)
    MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-2}"
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
    ROLLOUT_N="${ROLLOUT_N:-8}"
    VAL_N="${VAL_N:-4}"
    GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.4}"
fi

# 序列长度（minirank prompt 中位数 ~4300 tokens, p95 ~6700 tokens）
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-6144}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

# 生成参数
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.9}"
TOP_K="${TOP_K:--1}"

# ── 验证数据和模型 ───────────────────────────────────────────────────────
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

# ── 环境变量 ─────────────────────────────────────────────────────────────
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

# ── 创建输出目录（清理旧的 checkpoint 残留）─────────────────────────────
rm -rf "${save_path}/global_step_"* "${save_path}/latest_checkpointed_iteration.txt" 2>/dev/null || true
mkdir -p "${save_path}" "${SWANLAB_LOG_DIR}" 2>/dev/null || true

# ── 清理 Ray 残留 ───────────────────────────────────────────────────────
ray stop --force 2>/dev/null || true
rm -rf /tmp/ray 2>/dev/null || true
sleep 3

# ── 实验名称 ─────────────────────────────────────────────────────────────
JOB_NAME="${JOB_NAME:-grpo_ranking_v11_local_${N_GPUS}gpu_$(date +%Y%m%d_%H%M%S)}"
PROJECT_NAME="${PROJECT_NAME:-GRPO-Ranking}"

# ── 打印配置 ─────────────────────────────────────────────────────────────
echo "============================================================"
echo "  GRPO Ranking v11 本地训练（Pairwise Preference Reward）"
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
echo "  Reward   : pairwise_preference + rule_base + clean_bonus (v11)"
echo "  SwanLab  : ${SWANLAB_MODE}"
echo "  输出路径 : ${save_path}"
echo "============================================================"

# ── 启动训练（v11 Pairwise Preference Reward）─────────────────────────────
# 关键改动 vs v10:
#   - reward: pairwise preference {-1,0,+1} (v10 是 judge_norm 绝对打分, 无区分度)
#   - entropy_coeff: 0 → 0.02 (防止 entropy 坍缩)
#   - rollout_n: 4 → 8 (更多 rollout, pairwise 比较更充分)
#   - base_score: rule_score 确定性信号 (v10 是 judge_norm, 不可靠)
python -m verl.trainer.main_ppo \
    --config-name baseline_grpo \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.train_files="${train_data_path}" \
    data.val_files="${val_data_path}" \
    data.val_max_samples=100 \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.filter_overlong_prompts_workers=1 \
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
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    actor_rollout_ref.rollout.temperature=${TEMPERATURE} \
    actor_rollout_ref.rollout.top_p=${TOP_P} \
    actor_rollout_ref.rollout.top_k=${TOP_K} \
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF} \
    algorithm.norm_adv_by_std_in_grpo=True \
    trainer.total_epochs=30 \
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS} \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.resume_mode=disable \
    trainer.save_best_metric=val/test_score/mean \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.val_before_train=False \
    trainer.default_local_dir="${save_path}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${JOB_NAME}" \
    trainer.group_name="GRPO-ranking-v11-local" \
    "trainer.logger=[console,swanlab]"
