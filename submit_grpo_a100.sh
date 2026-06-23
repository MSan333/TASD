#!/bin/bash
# =============================================================================
# GRPO Ranking v4 A100 星云训练提交脚本
#
# 使用: bash submit_grpo_a100.sh
#
# 目标队列: lazada_llm_a100 (4× A100 80GB)
# =============================================================================

# ── Nebula 账号配置 ─────────────────────────────────────────────────────
QUEUE="lazada_llm_a100"
WORLD_SIZE=1  # 单节点 4 卡
OPENLM_TOKEN="OPENLM_539409_RjpprGplVLfSHXizAuEFtilWmFcpZFFE"
OSS_ACCESS_ID="${OSS_ACCESS_ID:?请设置 OSS_ACCESS_ID 环境变量}"
OSS_ACCESS_KEY="${OSS_ACCESS_KEY:?请设置 OSS_ACCESS_KEY 环境变量}"
OSS_ENDPOINT="oss-cn-hangzhou-zmf.aliyuncs.com"
OSS_BUCKET="lazada-ai-model"
CUSTOM_DOCKER_IMAGE="hub.docker.alibaba-inc.com/mdl/notebook_saved:guoshuaile.gsl_33v2_20260609193943"

# ── 训练脚本路径 ─────────────────────────────────────────────────────────
SCRIPT_PATH="nebula_scripts/grpo/grpo_ranking_gsl_v4_a100.sh"
CLUSTER_FILE="nebula_scripts/cluster_gpu_4.json"

# ── 任务命名 ──────────────────────────────────────────────────────────────
CURRENT_TIME=$(date +%Y%m%d_%H%M%S)
JOB_NAME="grpo_ranking_v4_a100_${CURRENT_TIME}"

echo "============================================================"
echo "提交 Nebula A100 任务"
echo "  脚本       : $SCRIPT_PATH"
echo "  节点数     : $WORLD_SIZE"
echo "  队列       : $QUEUE"
echo "  任务名     : $JOB_NAME"
echo "  Cluster    : $CLUSTER_FILE"
echo "  镜像       : $CUSTOM_DOCKER_IMAGE"
echo "============================================================"

# ── 构建 --env 参数 ─────────────────────────────────────────────────────
export OPENLM_TOKEN OSS_ACCESS_ID OSS_ACCESS_KEY

ENV_FLAGS="--env=OPENLM_TOKEN=${OPENLM_TOKEN}"
ENV_FLAGS="${ENV_FLAGS} --env=JOB_NAME=${JOB_NAME}"

# 传递关键训练超参（可选，脚本内已硬编码，这里仅做覆盖）
for var in DATASET LR MINI_BATCH_SIZE TRAIN_BATCH_SIZE ROLLOUT_N \
           MODEL_PATH KL_COEF TOTAL_TRAINING_STEPS VAL_N \
           PROJECT_NAME \
           OPENAI_API_KEY OPENAI_BASE_URL OPENAI_MODEL \
           SWANLAB_API_KEY SWANLAB_MODE; do
    val="${!var}"
    if [ -n "$val" ]; then
        ENV_FLAGS="${ENV_FLAGS} --env=${var}=${val}"
    fi
done

# ── 提交任务 ─────────────────────────────────────────────────────────────
SUBMIT_OUTPUT=$(nebulactl run mdl \
    --force \
    --engine=xdl \
    --queue=${QUEUE} \
    --entry=nebula_scripts/entry.py \
    --user_params="--script_path=${SCRIPT_PATH} --world_size=${WORLD_SIZE} --job_name=${JOB_NAME}" \
    --worker_count=${WORLD_SIZE} \
    --file.cluster_file=${CLUSTER_FILE} \
    --job_name=${JOB_NAME} \
    --access_id=${OSS_ACCESS_ID} \
    --access_key=${OSS_ACCESS_KEY} \
    ${ENV_FLAGS} \
    --custom_docker_image=${CUSTOM_DOCKER_IMAGE} \
    --requirements_file_name=requirements_nebula.txt \
    --oss_access_id=${OSS_ACCESS_ID} \
    --oss_access_key=${OSS_ACCESS_KEY} \
    --oss_bucket=${OSS_BUCKET} \
    --oss_endpoint=${OSS_ENDPOINT} 2>&1)

SUBMIT_EXIT=$?
echo "$SUBMIT_OUTPUT"

if [ $SUBMIT_EXIT -ne 0 ]; then
    echo "❌ 提交失败 (exit code: $SUBMIT_EXIT)"
    exit $SUBMIT_EXIT
fi

echo "✅ A100 任务提交成功"
echo "   任务名: $JOB_NAME"
echo "   队列:   $QUEUE"
echo "   SwanLab: https://swanlab.cn/@sansan/GRPO-Ranking-A100"
