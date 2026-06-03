#!/bin/bash
export PATH="/Users/xigaodi/Library/Python/3.9/bin:$PATH"
# =============================================================================
# GRPO Ranking 训练 - guoshauile.gsl Nebula 提交脚本
#
# 使用方式：
#   bash nebula_scripts/submit_grpo_ranking_gsl.sh --debug     # 单卡调试(5步)
#   bash nebula_scripts/submit_grpo_ranking_gsl.sh --dry-run   # 检查参数
#   bash nebula_scripts/submit_grpo_ranking_gsl.sh             # 4卡正式训练
# =============================================================================

# ── 加载凭据 ─────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/../.env"
if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
fi

# ── 模式判断 ─────────────────────────────────────────────────────────────
DEBUG_MODE=false
DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --debug)  DEBUG_MODE=true ;;
        --dry-run) DRY_RUN=true ;;
    esac
done

# ── Nebula 账号配置 ──────────────────────────────────────────────────────
QUEUE="lazada_llm_ad_h20"
WORLD_SIZE=1
OPENLM_TOKEN="${OPENLM_TOKEN:?OPENLM_TOKEN not set}"
OSS_ACCESS_ID="${OSS_ACCESS_ID:?OSS_ACCESS_ID not set}"
OSS_ACCESS_KEY="${OSS_ACCESS_KEY:?OSS_ACCESS_KEY not set}"
OSS_ENDPOINT="oss-cn-hangzhou-zmf.aliyuncs.com"
OSS_BUCKET="lazada-ai-model"
SCRIPT_PATH="nebula_scripts/grpo/grpo_ranking_gsl.sh"
CUSTOM_DOCKER_IMAGE="${CUSTOM_DOCKER_IMAGE:-hub.docker.alibaba-inc.com/mdl/notebook_saved:loujieming.ljm_yueqiu_sdpo_env_torch260_20260324155942}"
OSS_ROOT="/data/oss_bucket_0/ad/guoshauile.gsl"

# ── 根据模式设置参数 ─────────────────────────────────────────────────────
DATASET="${DATASET:-ranking}"
MODEL_PATH="${MODEL_PATH:-/data/oss_bucket_0/ad/loujieming.ljm/base_models/Qwen3-8B}"

if [ "$DEBUG_MODE" = true ]; then
    PROJECT_NAME="GRPO-Ranking-GSL-Debug"
    CLUSTER_FILE="nebula_scripts/cluster_gpu_1.json"
    N_GPUS=1
    LR="${LR:-1e-6}"
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
    MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-4}"
    ROLLOUT_N="${ROLLOUT_N:-2}"
    VAL_N=2
    KL_COEF="${KL_COEF:-0.05}"
    TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-5}"
else
    PROJECT_NAME="GRPO-Ranking-GSL"
    CLUSTER_FILE="nebula_scripts/cluster_gpu_4.json"
    N_GPUS=4
    LR="${LR:-5e-7}"
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
    MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-16}"
    ROLLOUT_N="${ROLLOUT_N:-8}"
    VAL_N=16
    KL_COEF="${KL_COEF:-0.05}"
    TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-500}"
fi

# ── 任务命名 ──────────────────────────────────────────────────────────────
CURRENT_TIME=$(date +%Y%m%d_%H%M%S)
MODEL_SHORT=$(basename "$MODEL_PATH")
if [ "$DEBUG_MODE" = true ]; then
    JOB_NAME="GRPO-debug-1gpu-${CURRENT_TIME}"
else
    LR_TAG=$(echo "$LR" | tr '-' '_')
    JOB_NAME="GRPO-ranking-gsl-lr${LR_TAG}-mbs${MINI_BATCH_SIZE}-k${ROLLOUT_N}-kl${KL_COEF}-${MODEL_SHORT}-${CURRENT_TIME}"
fi

echo "============================================================"
if [ "$DEBUG_MODE" = true ]; then
    echo "GRPO Ranking 单卡调试 (1×H20)"
else
    echo "GRPO Ranking 正式训练 (4×H20)"
fi
echo "  数据集      : ${OSS_ROOT}/data/${DATASET}/"
echo "  模型        : ${MODEL_PATH}"
echo "  输出        : ${OSS_ROOT}/result/${JOB_NAME}"
echo "  ────────────────────────────────────"
echo "  GPU         : ${N_GPUS}"
echo "  LR          : ${LR}"
echo "  Batch size  : ${TRAIN_BATCH_SIZE}"
echo "  Mini batch  : ${MINI_BATCH_SIZE}"
echo "  Rollout K   : ${ROLLOUT_N}"
echo "  Val N       : ${VAL_N}"
echo "  KL coef     : ${KL_COEF}"
echo "  Steps       : ${TOTAL_TRAINING_STEPS}"
echo "  Job name    : ${JOB_NAME}"
echo "============================================================"

if [ "$DRY_RUN" = true ]; then
    echo "[dry-run] 以上为将要提交的参数，不实际提交"
    exit 0
fi

SUBMIT_OUTPUT=$(nebulactl run mdl \
    --force \
    --engine=xdl \
    --queue=${QUEUE} \
    --entry=nebula_scripts/entry.py \
    --user_params="--script_path=${SCRIPT_PATH} --world_size=${WORLD_SIZE} --job_name=${JOB_NAME} --env=PROJECT_NAME=${PROJECT_NAME} --env=JOB_NAME=${JOB_NAME} --env=DATASET=${DATASET} --env=MODEL_PATH=${MODEL_PATH} --env=LR=${LR} --env=MINI_BATCH_SIZE=${MINI_BATCH_SIZE} --env=TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} --env=ROLLOUT_N=${ROLLOUT_N} --env=KL_COEF=${KL_COEF} --env=TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS} --env=N_GPUS=${N_GPUS} --env=VAL_N=${VAL_N}" \
    --worker_count=${WORLD_SIZE} \
    --file.cluster_file=${CLUSTER_FILE} \
    --job_name=${JOB_NAME} \
    --access_id=${access_id} \
    --access_key=${access_key} \
    --env=OPENLM_TOKEN=${OPENLM_TOKEN} \
    --env=SWANLAB_API_KEY=${SWANLAB_API_KEY} \
    $([ -n "$CUSTOM_DOCKER_IMAGE" ] && echo "--custom_docker_image=${CUSTOM_DOCKER_IMAGE}" || echo "--algo_name=pytorch260") \
    --requirements_file_name=requirements_nebula.txt \
    --oss_access_id=${OSS_ACCESS_ID} \
    --oss_access_key=${OSS_ACCESS_KEY} \
    --oss_bucket=${OSS_BUCKET} \
    --oss_endpoint=${OSS_ENDPOINT} 2>&1)
SUBMIT_EXIT=$?
echo "$SUBMIT_OUTPUT"
if [ $SUBMIT_EXIT -ne 0 ]; then
    echo "❌ 提交失败 (exit code: $SUBMIT_EXIT)"
else
    echo "✅ 提交成功: ${JOB_NAME}"
fi
