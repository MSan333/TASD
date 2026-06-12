#!/bin/bash
# =============================================================================
# 环境检查提交脚本
# 提交 check_env.sh 到 Nebula 验证环境（不跑训练，排到后几分钟跑完）
#
# 用法: bash submit_check_env.sh
# =============================================================================

bash nebula_scripts/submit_job.sh \
    nebula_scripts/grpo/check_env.sh \
    1 \
    lazada_llm_ad_h20
