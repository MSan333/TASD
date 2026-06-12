#!/usr/bin/env bash
# =============================================================================
# 环境检查脚本 — 提交到 Nebula 验证环境，不跑训练
# 走正常 GPU 队列（4卡 H20），排到后几分钟跑完
#
# 用法: bash submit_check_env.sh
# =============================================================================
set -x

echo "============================================================"
echo "环境检查 $(date)"
echo "============================================================"

PASS=0
FAIL=0
check() {
    if eval "$1"; then
        echo "  ✅ $2"
        PASS=$((PASS + 1))
    else
        echo "  ❌ $2"
        FAIL=$((FAIL + 1))
    fi
}

# ── 1. Python 环境 ──────────────────────────────────────────────────
echo ""
echo "── 1. Python 环境 ──"
echo "  which python: $(which python)"
echo "  python --version: $(python --version 2>&1)"

python -c "import sys; assert sys.version_info >= (3, 10), f'Python {sys.version} < 3.10'"
check "[ $? -eq 0 ]" "Python >= 3.10"

# ── 2. GPU 信息 ─────────────────────────────────────────────────────
echo ""
echo "── 2. GPU 信息 ──"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
check "[ $? -eq 0 ]" "nvidia-smi 可用"
echo "  CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-未设置}"

# ── 3. 关键 Python 包导入 ───────────────────────────────────────────
echo ""
echo "── 3. 关键 Python 包 ──"

python -c "import torch; print(f'  torch={torch.__version__}, cuda={torch.cuda.is_available()}')"
check "[ $? -eq 0 ]" "torch"

python -c "import vllm; print(f'  vllm={vllm.__version__}')"
check "[ $? -eq 0 ]" "vllm"

python -c "import ray; print(f'  ray={ray.__version__}')"
check "[ $? -eq 0 ]" "ray"

python -c "import transformers; print(f'  transformers={transformers.__version__}')"
check "[ $? -eq 0 ]" "transformers"

python -c "import datasets; print(f'  datasets={datasets.__version__}')"
check "[ $? -eq 0 ]" "datasets"

python -c "import hydra; print(f'  hydra={hydra.__version__}')"
check "[ $? -eq 0 ]" "hydra"

python -c "import omegaconf; print(f'  omegaconf={omegaconf.__version__}')"
check "[ $? -eq 0 ]" "omegaconf"

python -c "import swanlab; print(f'  swanlab={swanlab.__version__}')"
check "[ $? -eq 0 ]" "swanlab"

python -c "import openai; print(f'  openai={openai.__version__}')"
check "[ $? -eq 0 ]" "openai (LLM Judge)"

python -c "import peft; print(f'  peft={peft.__version__}')"
check "[ $? -eq 0 ]" "peft"

python -c "import tensordict; print(f'  tensordict={tensordict.__version__}')"
check "[ $? -eq 0 ]" "tensordict"

python -c "import liger_kernel; print('  liger_kernel OK')"
check "[ $? -eq 0 ]" "liger_kernel"

# ── 4. 项目代码安装 ─────────────────────────────────────────────────
echo ""
echo "── 4. 项目代码 ──"
echo "  当前目录: $(pwd)"
echo "  安装项目 (pip install -e .)..."
pip install -e . --no-deps --no-build-isolation --quiet 2>&1 | tail -3

python -c "import verl; print('  verl 导入成功')"
check "[ $? -eq 0 ]" "verl (pip install -e . 后)"

ls -la verl/trainer/main_ppo.py 2>/dev/null
check "[ $? -eq 0 ]" "verl/trainer/main_ppo.py 存在"

ls -la verl/utils/reward_score/feedback/__init__.py 2>/dev/null
check "[ $? -eq 0 ]" "reward 函数文件存在"

# ── 5. 数据文件检查 ─────────────────────────────────────────────────
echo ""
echo "── 5. 数据文件 ──"
OSS_ROOT="/data/oss_bucket_0/ad/guoshauile.gsl"

ls -lh "${OSS_ROOT}/data/ranking/train.parquet" 2>/dev/null
check "[ $? -eq 0 ]" "训练数据 train.parquet"

ls -lh "${OSS_ROOT}/data/ranking/test.parquet" 2>/dev/null
check "[ $? -eq 0 ]" "验证数据 test.parquet"

# ── 6. 模型文件检查 ─────────────────────────────────────────────────
echo ""
echo "── 6. 模型文件 ──"
MODEL_PATH="${OSS_ROOT}/model/qwen3-8b"

ls -lh "${MODEL_PATH}/config.json" 2>/dev/null
check "[ $? -eq 0 ]" "config.json"

SAFETENSOR_COUNT=$(ls "${MODEL_PATH}/"*.safetensors 2>/dev/null | wc -l)
echo "  safetensors 文件数: ${SAFETENSOR_COUNT}"
ls "${MODEL_PATH}/"*.safetensors 2>/dev/null | head -5
check "[ ${SAFETENSOR_COUNT} -gt 0 ]" "模型权重文件 (${SAFETENSOR_COUNT} 个)"

ls -lh "${MODEL_PATH}/tokenizer.json" 2>/dev/null || ls -lh "${MODEL_PATH}/tokenizer_config.json" 2>/dev/null
check "[ $? -eq 0 ]" "tokenizer 文件"

# ── 7. Hydra 配置检查 ───────────────────────────────────────────────
echo ""
echo "── 7. Hydra 配置 ──"
python -c "
from hydra import compose, initialize_config_dir
import os, sys

# 尝试多个可能的 config 路径
candidates = [
    os.path.abspath('verl/trainer/config/ppo'),
    os.path.abspath('verl/trainer/config'),
]
for config_dir in candidates:
    if os.path.isdir(config_dir):
        print(f'  找到 config 目录: {config_dir}')
        break
else:
    print('  ❌ 未找到 config 目录')
    sys.exit(1)
"
check "[ $? -eq 0 ]" "Hydra config 目录"

# ── 8. 环境变量汇总 ─────────────────────────────────────────────────
echo ""
echo "── 8. 关键环境变量 ──"
echo "  PYTHONPATH    = ${PYTHONPATH}"
echo "  CUDA_VISIBLE  = ${CUDA_VISIBLE_DEVICES}"
echo "  JOB_NAME      = ${JOB_NAME}"
echo "  PATH(python)  = $(which python)"
echo "  RAY 状态      = $(ray status 2>&1 | head -3)"

# ── 汇总 ────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "检查结果: ${PASS} 通过, ${FAIL} 失败"
if [ $FAIL -eq 0 ]; then
    echo "🎉 全部通过！可以放心提交训练任务"
    echo "   运行: bash submit_grpo.sh"
else
    echo "⚠️  有 ${FAIL} 项失败"
    echo "   请查看上方 ❌ 标记，修复后重新提交"
fi
echo "============================================================"
