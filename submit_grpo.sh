#!/bin/bash
# =============================================================================
# GRPO Ranking 星云训练提交脚本
#
# ★ 使用前必须先设置以下环境变量（替换为你的真实密钥）★
# 建议写入 ~/.bashrc 避免每次手动设置：
#
#   # ── 星云/Nebula 账号（必须）──
#   export OPENLM_TOKEN="你的openlm_token"
#   export OSS_ACCESS_ID="你的oss_access_id"
#   export OSS_ACCESS_KEY="你的oss_access_key"
#
#   # ── LLM Judge API Key（可选，Reward 打分用）──
#   export OPENAI_API_KEY="你的dashscope_api_key"
#
# 设置好后运行:
#   bash submit_grpo.sh
# =============================================================================

export OPENLM_TOKEN="${OPENLM_TOKEN:?请先执行: export OPENLM_TOKEN=\"你的openlm_token\"}"
export OSS_ACCESS_ID="${OSS_ACCESS_ID:?请先执行: export OSS_ACCESS_ID=\"你的oss_access_id\"}"
export OSS_ACCESS_KEY="${OSS_ACCESS_KEY:?请先执行: export OSS_ACCESS_KEY=\"你的oss_access_key\"}"

bash nebula_scripts/submit_job.sh \
    nebula_scripts/grpo/grpo_ranking_gsl.sh \
    1 \
    lazada_llm_ad_h20
