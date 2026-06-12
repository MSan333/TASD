#!/bin/bash
# =============================================================================
# TASD Nebula 任务提交脚本
#
# 使用前：需要先在脚本下方「密钥配置」区域填入你的密钥（仅本地使用，不要提交到 git）
# 或者 export 环境变量后运行。
#
# 使用方式：
#   bash nebula_scripts/submit_job.sh <script_path> [world_size] [KEY=VALUE ...]
#
# 示例：
#   # 最简提交（单节点 4 卡，所有超参使用脚本默认值）
#   bash nebula_scripts/submit_job.sh nebula_scripts/grpo/grpo_ranking_gsl.sh
#
#   # 覆盖超参
#   bash nebula_scripts/submit_job.sh nebula_scripts/grpo/grpo_ranking_gsl.sh 1 \
#       LR=5e-6 TRAIN_BATCH_SIZE=64
#
#   # 多节点（2 节点 × 8 卡 = 16 卡）
#   bash nebula_scripts/submit_job.sh nebula_scripts/grpo/grpo_ranking_gsl.sh 2
# =============================================================================
set -euo pipefail

# ── 密钥配置 ──────────────────────────────────────────────────────────────
# ⚠️  请在本地填入你的密钥，不要提交到 git！
# 可以直接修改下面的值，也可以通过 export 环境变量覆盖。
# 详见 ENV_SETUP.md
OPENLM_TOKEN="${OPENLM_TOKEN:-__FILL_YOUR_OPENLM_TOKEN__}"
OSS_ACCESS_ID="${OSS_ACCESS_ID:-__FILL_YOUR_OSS_ACCESS_ID__}"
OSS_ACCESS_KEY="${OSS_ACCESS_KEY:-__FILL_YOUR_OSS_ACCESS_KEY__}"
SWANLAB_API_KEY="${SWANLAB_API_KEY:-__FILL_YOUR_SWANLAB_API_KEY__}"

# 检查密钥是否已配置
for var_name in OPENLM_TOKEN OSS_ACCESS_ID OSS_ACCESS_KEY; do
    eval "val=\$var_name"
    if [[ "$val" == __FILL_* ]]; then
        echo "❌ 请先配置 ${var_name}！"
        echo "   方式1: 直接编辑本脚本的「密钥配置」区域"
        echo "   方式2: export ${var_name}=\"你的值\" 后运行"
        echo "   详见 ENV_SETUP.md"
        exit 1
    fi
done

# ── Nebula 平台配置 ───────────────────────────────────────────────────────
QUEUE="lazada_llm_ad_h20"
OSS_ENDPOINT="oss-cn-hangzhou-zmf.aliyuncs.com"
OSS_BUCKET="lazada-ai-model"
CUSTOM_DOCKER_IMAGE="${CUSTOM_DOCKER_IMAGE:-hub.docker.alibaba-inc.com/mdl/notebook_saved:guoshuaile.gsl_33v2_20260609193943}"

# ── 参数解析 ──────────────────────────────────────────────────────────────
script_dir_path="${1:-nebula_scripts/grpo/grpo_ranking_gsl.sh}"
WORLD_SIZE="${2:-1}"

# 收集第3个参数之后的 KEY=VALUE 对作为环境变量透传
ENV_ARGS=""
shift 2 2>/dev/null || true
for arg in "$@"; do
    if [[ "$arg" == *"="* ]]; then
        ENV_ARGS="${ENV_ARGS} --env ${arg}"
        echo "  透传环境变量: ${arg}"
    fi
done

# ── 根据节点数选择 cluster 配置文件 ──────────────────────────────────────
if [ "$WORLD_SIZE" -gt 1 ]; then
    CLUSTER_FILE="nebula_scripts/cluster.json"          # 8 GPU × N 节点
else
    CLUSTER_FILE="nebula_scripts/cluster_gpu_4.json"    # 单节点 4 GPU（省钱）
fi

# ── 任务命名 ──────────────────────────────────────────────────────────────
CURRENT_TIME=$(date +%Y%m%d_%H%M%S)
JOB_NAME="$(basename "${script_dir_path%.sh}")_${CURRENT_TIME}"

# ── 提交参数（通过 entry.py 的 --env 透传环境变量到训练脚本）────────────
options="--script_path=${script_dir_path} --world_size=${WORLD_SIZE} --job_name=${JOB_NAME}${ENV_ARGS}"

echo "============================================================"
echo "提交 Nebula 任务"
echo "  脚本       : $script_dir_path"
echo "  节点数     : $WORLD_SIZE"
echo "  队列       : $QUEUE"
echo "  任务名     : $JOB_NAME"
echo "  Cluster    : $CLUSTER_FILE"
echo "  镜像       : $CUSTOM_DOCKER_IMAGE"
echo "============================================================"

SUBMIT_OUTPUT=$(nebulactl run mdl \
    --force \
    --engine=xdl \
    --queue=${QUEUE} \
    --entry=nebula_scripts/entry.py \
    --user_params="${options}" \
    --worker_count=${WORLD_SIZE} \
    --file.cluster_file=${CLUSTER_FILE} \
    --job_name=${JOB_NAME} \
    --access_id=${OSS_ACCESS_ID} \
    --access_key=${OSS_ACCESS_KEY} \
    --env=OPENLM_TOKEN=${OPENLM_TOKEN} \
    --env=SWANLAB_API_KEY=${SWANLAB_API_KEY} \
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
else
    echo "✅ 提交成功"
fi
