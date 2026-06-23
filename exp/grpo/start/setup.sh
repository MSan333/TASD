#!/usr/bin/env bash
# =============================================================================
# GRPO 本地环境一键初始化（基于 uv）
#
# 功能：
#   1. 安装 uv（如果未安装）
#   2. 创建 Python 3.10 虚拟环境
#   3. 安装所有训练依赖（torch、vllm、ray、transformers 等）
#   4. 以开发模式安装 verl 项目
#
# 使用方式：
#   bash exp/grpo/start/setup.sh
#
# 安装完成后激活环境：
#   source .venv/bin/activate
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${PROJECT_ROOT}"
echo "项目根目录: ${PROJECT_ROOT}"

# ── Step 1: 安装 uv ─────────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo "📦 安装 uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
    echo "✅ uv 已安装: $(uv --version)"
else
    echo "✅ uv 已存在: $(uv --version)"
fi

# ── Step 2: 创建虚拟环境 ────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
    echo "🐍 创建 Python 3.10 虚拟环境..."
    # 优先使用 conda python3.10.13（星云/notebook 镜像已有完整 CUDA 环境）
    if [ -x "/opt/conda/envs/python3.10.13/bin/python" ]; then
        uv venv .venv --python /opt/conda/envs/python3.10.13/bin/python --seed
        echo "  基于 conda python3.10.13 创建"
    else
        uv venv --python 3.10 --seed
        echo "  基于系统 Python 3.10 创建"
    fi
    echo "✅ 虚拟环境已创建: .venv/"
else
    echo "✅ 虚拟环境已存在: .venv/"
fi

# ── Step 3: 安装 PyTorch（仅在 .venv 里没有 torch 时才安装）─────────────
if ! .venv/bin/python -c "import torch" 2>/dev/null; then
    echo "🔥 安装 PyTorch..."
    uv pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu124
else
    echo "✅ PyTorch 已存在: $(.venv/bin/python -c 'import torch; print(torch.__version__)')"
fi

# ── Step 4: 安装项目依赖 ────────────────────────────────────────────────
echo "📦 安装项目依赖..."
uv pip install -r requirements.txt

# ── Step 5: 以开发模式安装 verl ──────────────────────────────────────────
echo "📦 安装 verl (开发模式)..."
uv pip install -e . --no-deps --no-build-isolation

# ── Step 6: 安装额外依赖 ────────────────────────────────────────────────
echo "📦 安装额外依赖..."
uv pip install swanlab openlm-hub filelock "huggingface_hub<0.28"

# ── Step 7: 验证安装 ────────────────────────────────────────────────────
echo ""
echo "🔍 验证安装..."
.venv/bin/python -c "
import importlib
deps = ['torch', 'ray', 'vllm', 'transformers', 'huggingface_hub', 'peft', 'swanlab', 'verl']
all_ok = True
for d in deps:
    try:
        m = importlib.import_module(d)
        v = getattr(m, '__version__', 'ok')
        print(f'  ✅ {d}=={v}')
    except ImportError:
        print(f'  ❌ {d} 缺失')
        all_ok = False

import torch
print(f'  🖥️  CUDA 可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  🖥️  GPU 数量: {torch.cuda.device_count()}')
    print(f'  🖥️  GPU 型号: {torch.cuda.get_device_name(0)}')

if all_ok:
    print()
    print('🎉 环境安装成功！')
    print('   激活环境: source .venv/bin/activate')
    print('   本地训练: bash exp/grpo/start/train.sh')
    print('   星云提交: bash exp/grpo/start/submit_nebula.sh')
else:
    print()
    print('⚠️  部分依赖缺失，请检查上面的错误信息')
"
