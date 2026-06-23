#!/usr/bin/env bash
# =============================================================================
# GRPO Ranking 星云提交脚本 — uv 环境版
#
# 与 submit_nebula.sh 的区别：
#   - 使用自定义 Docker 镜像（通过 CUSTOM_DOCKER_IMAGE 指定）
#   - 训练脚本为 grpo_ranking_gsl_v7_uv.sh（uv 激活环境）
#   - 默认单节点 4 卡
#
# ⚠️ 本文件包含密钥配置，仅供本地使用，不要提交到 git！
#
# 使用方式：
#   # 1. 先构建并推送镜像（在其他地方完成）
#   #    docker build -f Dockerfile.nebula -t <your_image_tag> .
#   #    docker push <your_image_tag>
#
#   # 2. 提交训练任务
#   CUSTOM_DOCKER_IMAGE=<your_image_tag> bash exp/grpo/start/submit_nebula_uv.sh
#
#   # 覆盖超参
#   CUSTOM_DOCKER_IMAGE=<your_image_tag> bash exp/grpo/start/submit_nebula_uv.sh LR=5e-6 TOTAL_TRAINING_STEPS=200
#
#   # 多节点
#   CUSTOM_DOCKER_IMAGE=<your_image_tag> WORLD_SIZE=2 bash exp/grpo/start/submit_nebula_uv.sh
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${PROJECT_ROOT}"

# ── 密钥配置 ────────────────────────────────────────────────────────────
# OPENLM_TOKEN / OSS 凭证从环境变量读取（敏感信息，不要提交到 git）
# SWANLAB_API_KEY 可明文，无安全风险
export OPENLM_TOKEN="${OPENLM_TOKEN:?请设置 OPENLM_TOKEN 环境变量}"
export OSS_ACCESS_ID="${OSS_ACCESS_ID:?请设置 OSS_ACCESS_ID 环境变量}"
export OSS_ACCESS_KEY="${OSS_ACCESS_KEY:?请设置 OSS_ACCESS_KEY 环境变量}"
export SWANLAB_API_KEY="o4MGQAOSX8rGztH69Jj5P"

# ── 自定义镜像（必须设置）──────────────────────────────────────────────
if [ -z "${CUSTOM_DOCKER_IMAGE:-}" ]; then
    echo "❌ 请设置 CUSTOM_DOCKER_IMAGE 环境变量"
    echo ""
    echo "用法："
    echo "  CUSTOM_DOCKER_IMAGE=<your_image_tag> bash $0"
    echo ""
    echo "示例："
    echo "  CUSTOM_DOCKER_IMAGE=hub.docker.alibaba-inc.com/mdl/tasd_grpo:latest bash $0"
    exit 1
fi
export CUSTOM_DOCKER_IMAGE

# ── 训练配置 ──────────────────────────────────────────────────────────
SCRIPT_PATH="nebula_scripts/grpo/grpo_ranking_gsl_v7_uv.sh"
WORLD_SIZE="${WORLD_SIZE:-1}"

# 收集 KEY=VALUE 超参
ENV_EXTRA=""
for arg in "$@"; do
    if [[ "$arg" == *"="* ]]; then
        ENV_EXTRA="${ENV_EXTRA} ${arg}"
    fi
done

# ── 调用通用提交脚本 ──────────────────────────────────────────────────
echo "🚀 提交 GRPO Ranking (uv 环境版) 到星云..."
echo "  镜像: ${CUSTOM_DOCKER_IMAGE}"
echo "  脚本: ${SCRIPT_PATH}"
echo "  节点: ${WORLD_SIZE}"
bash nebula_scripts/submit_job.sh "${SCRIPT_PATH}" "${WORLD_SIZE}" ${ENV_EXTRA}
