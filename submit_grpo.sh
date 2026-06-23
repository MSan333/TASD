#!/bin/bash
# =============================================================================
# GRPO Ranking 星云训练提交脚本
# 使用: bash submit_grpo.sh
# =============================================================================

export OPENLM_TOKEN="OPENLM_539409_RjpprGplVLfSHXizAuEFtilWmFcpZFFE"
export OSS_ACCESS_ID="${OSS_ACCESS_ID:?请设置 OSS_ACCESS_ID 环境变量}"
export OSS_ACCESS_KEY="${OSS_ACCESS_KEY:?请设置 OSS_ACCESS_KEY 环境变量}"

bash nebula_scripts/submit_job.sh \
    nebula_scripts/grpo/grpo_ranking_gsl_v4.sh \
    1 \
    lazada_llm_ad_h20
