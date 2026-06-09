#!/usr/bin/env bash
# =============================================================================
# TASD 环境自动检测与初始化脚本
#
# 自动检测当前运行环境（新镜像 / 老镜像 / Notebook 本地），
# 激活正确的 Python 环境并安装缺失依赖。
#
# 使用方式（必须用 source 加载）:
#
#   source setup_env.sh
#
# 支持的运行环境:
#   1. 新镜像 (Docker/星云) — 所有依赖已预装，直接可用
#   2. 老镜像 (Docker/星云) — 基础环境，增量安装 TASD 依赖
#   3. Notebook 本地         — conda 环境 (python3.10 或 sdpo_env)
#
# =============================================================================

# ── 颜色定义 ────────────────────────────────────────────────────────────
_GREEN='\033[0;32m'
_YELLOW='\033[1;33m'
_RED='\033[0;31m'
_BLUE='\033[0;34m'
_NC='\033[0m'

# ── 镜像地址 ────────────────────────────────────────────────────────────
export DOCKER_IMAGE_NEW="hub.docker.alibaba-inc.com/mdl/notebook_saved:guoshuaile.gsl_33v2_20260609193943"
export DOCKER_IMAGE_OLD="hub.docker.alibaba-inc.com/mdl/notebook_saved:guoshuaile.gsl_llm_rl_20260605230018"

# ── 默认使用新镜像（星云提交时使用）────────────────────────────────────
export CUSTOM_DOCKER_IMAGE="${CUSTOM_DOCKER_IMAGE:-${DOCKER_IMAGE_NEW}}"

# ── 项目根目录 ──────────────────────────────────────────────────────────
TASD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${TASD_ROOT}:${PYTHONPATH:-}"

# ── 阿里源 (内网加速) ──────────────────────────────────────────────────
PIP_MIRROR="--trusted-host mirrors.aliyun.com -i http://mirrors.aliyun.com/pypi/simple/"

# =============================================================================
# 环境检测逻辑
# =============================================================================

detect_environment() {
    # 检测是否在 Docker 容器中
    local in_docker=false
    if [ -f /.dockerenv ] || grep -qsm1 'docker\|containerd' /proc/1/cgroup 2>/dev/null; then
        in_docker=true
    fi

    if [ "${in_docker}" = true ]; then
        # 在容器中 — 通过镜像标签区分新老镜像
        # 检查新镜像特征：新镜像预装了 swanlab + verl + vllm
        if python -c "import swanlab, vllm, ray" 2>/dev/null; then
            echo "new_docker"
        else
            echo "old_docker"
        fi
    else
        # 不在容器中 — Notebook 本地环境
        echo "notebook"
    fi
}

# =============================================================================
# Python 环境激活
# =============================================================================

activate_python_env() {
    local env_type="$1"

    case "${env_type}" in
        new_docker)
            echo -e "${_GREEN}✅ 新镜像环境 — 依赖已预装，无需额外安装${_NC}"
            # 尝试激活容器内的 conda 环境
            if [ -d "/opt/conda/envs/python3.10/bin" ]; then
                export PATH="/opt/conda/envs/python3.10/bin:${PATH}"
            elif [ -d "/opt/conda/envs/sdpo_env/bin" ]; then
                export PATH="/opt/conda/envs/sdpo_env/bin:${PATH}"
            fi
            ;;

        old_docker)
            echo -e "${_YELLOW}⚠️  老镜像环境 — 检查并增量安装缺失依赖${_NC}"
            # 激活 conda 环境
            if [ -d "/opt/conda/envs/sdpo_env/bin" ]; then
                export PATH="/opt/conda/envs/sdpo_env/bin:${PATH}"
                echo -e "${_BLUE}   使用 conda 环境: sdpo_env${_NC}"
            elif [ -d "/opt/conda/envs/python3.10/bin" ]; then
                export PATH="/opt/conda/envs/python3.10/bin:${PATH}"
                echo -e "${_BLUE}   使用 conda 环境: python3.10${_NC}"
            fi
            # 增量安装
            _install_incremental
            ;;

        notebook)
            echo -e "${_BLUE}📓 Notebook 本地环境${_NC}"
            # 按优先级尝试激活 conda 环境
            if [ -d "/opt/conda/envs/python3.10/bin" ]; then
                export PATH="/opt/conda/envs/python3.10/bin:${PATH}"
                echo -e "${_GREEN}   激活 conda 环境: python3.10${_NC}"
            elif [ -d "/opt/conda/envs/sdpo_env/bin" ]; then
                export PATH="/opt/conda/envs/sdpo_env/bin:${PATH}"
                echo -e "${_GREEN}   激活 conda 环境: sdpo_env${_NC}"
            else
                echo -e "${_YELLOW}   未找到预配置的 conda 环境，使用系统 Python${_NC}"
                echo -e "${_YELLOW}   建议: conda create -n python3.10 python=3.10 && conda activate python3.10${_NC}"
            fi
            ;;
    esac
}

# =============================================================================
# 增量安装（老镜像/首次本地开发）
# =============================================================================

_install_incremental() {
    local need_install=false

    # 检测核心依赖是否缺失
    for pkg in swanlab vllm ray hydra; do
        if ! python -c "import ${pkg}" 2>/dev/null; then
            echo -e "${_YELLOW}   缺失: ${pkg}${_NC}"
            need_install=true
        fi
    done

    if [ "${need_install}" = true ]; then
        echo -e "${_BLUE}   正在安装缺失依赖...${_NC}"
        pip install -r "${TASD_ROOT}/requirements_flex.txt" ${PIP_MIRROR} --quiet 2>&1 | tail -3
        echo -e "${_GREEN}   依赖安装完成${_NC}"
    fi

    # 安装 TASD 项目（editable mode）
    pip install -e "${TASD_ROOT}" --no-deps --no-build-isolation --quiet 2>/dev/null || true

    # 检查 flash-attn
    if ! python -c "import flash_attn" 2>/dev/null; then
        echo -e "${_YELLOW}   flash-attn 未安装，正在编译安装（约 10-20 分钟）...${_NC}"
        echo -e "${_YELLOW}   如需跳过，按 Ctrl+C 中断后设置 SKIP_FLASH_ATTN=1 重新 source${_NC}"
        if [ "${SKIP_FLASH_ATTN:-0}" != "1" ]; then
            MAX_JOBS=${MAX_JOBS:-4} FLASH_ATTENTION_FORCE_BUILD=TRUE \
                pip install flash-attn --no-build-isolation ${PIP_MIRROR} --quiet 2>&1 | tail -3
        else
            echo -e "${_YELLOW}   已跳过 flash-attn 安装${_NC}"
        fi
    fi
}

# =============================================================================
# 通用环境变量
# =============================================================================

setup_common_env() {
    # vLLM 配置
    unset VLLM_ATTENTION_BACKEND
    export VLLM_USE_V1=1
    export VLLM_LOGGING_LEVEL=WARN

    # Ray 配置
    export RAY_memory_monitor_refresh_ms=0

    # Wandb 关闭（使用 SwanLab）
    export WANDB_MODE=offline

    # PyTorch 配置
    export TORCH_WARN_ACCUMULATE_GRAD_STREAM=0

    # SwanLab 配置
    export SWANLAB_MODE="${SWANLAB_MODE:-cloud}"
    export SWANLAB_API_KEY="${SWANLAB_API_KEY:-o4MGQAOSX8rGztH69Jj5P}"

    # DashScope (LLM Judge)
    export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-93bf8a433943448bad6611ca5532a113}"
    export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
    export OPENAI_MODEL="${OPENAI_MODEL:-qwen3.6-max-preview}"
}

# =============================================================================
# 环境验证
# =============================================================================

verify_environment() {
    echo ""
    echo -e "${_BLUE}── 环境验证 ──────────────────────────────────────────${_NC}"

    # Python 版本
    local python_ver
    python_ver=$(python --version 2>&1)
    echo -e "   Python     : ${python_ver}"

    # PyTorch
    local torch_ver
    torch_ver=$(python -c "import torch; print(f'{torch.__version__} (CUDA {torch.version.cuda})')" 2>/dev/null || echo "❌ 未安装")
    echo -e "   PyTorch    : ${torch_ver}"

    # GPU
    local gpu_count
    gpu_count=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
    if [ "${gpu_count}" -gt 0 ]; then
        local gpu_name
        gpu_name=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "unknown")
        echo -e "   GPU        : ${_GREEN}${gpu_count}x ${gpu_name}${_NC}"
    else
        echo -e "   GPU        : ${_YELLOW}未检测到 GPU${_NC}"
    fi

    # 核心依赖检查
    local all_ok=true
    for pkg in ray vllm swanlab; do
        if python -c "import ${pkg}" 2>/dev/null; then
            local pkg_ver
            pkg_ver=$(python -c "import ${pkg}; print(${pkg}.__version__)" 2>/dev/null || echo "?")
            echo -e "   ${pkg}$(printf '%*s' $((10 - ${#pkg})) '')  : ${_GREEN}${pkg_ver}${_NC}"
        else
            echo -e "   ${pkg}$(printf '%*s' $((10 - ${#pkg})) '')  : ${_RED}❌ 未安装${_NC}"
            all_ok=false
        fi
    done

    # flash-attn
    if python -c "import flash_attn" 2>/dev/null; then
        local fa_ver
        fa_ver=$(python -c "import flash_attn; print(flash_attn.__version__)" 2>/dev/null || echo "?")
        echo -e "   flash-attn : ${_GREEN}${fa_ver}${_NC}"
    else
        echo -e "   flash-attn : ${_YELLOW}未安装（非必须，但推荐）${_NC}"
    fi

    echo -e "${_BLUE}────────────────────────────────────────────────────────${_NC}"

    # 镜像信息
    echo ""
    echo -e "${_BLUE}── 镜像配置 ──────────────────────────────────────────${_NC}"
    echo -e "   新镜像 : ${DOCKER_IMAGE_NEW}"
    echo -e "   老镜像 : ${DOCKER_IMAGE_OLD}"
    echo -e "   当前   : ${CUSTOM_DOCKER_IMAGE}"
    echo -e "${_BLUE}────────────────────────────────────────────────────────${_NC}"

    if [ "${all_ok}" = true ]; then
        echo ""
        echo -e "${_GREEN}✅ 环境就绪，可以开始训练${_NC}"
        echo ""
        echo -e "   训练命令示例:"
        echo -e "     ${_BLUE}source exp/grpo/env.sh && bash run_local_grpo_ranking.sh${_NC}"
        echo ""
        echo -e "   星云提交示例:"
        echo -e "     ${_BLUE}source exp/grpo/env.sh${_NC}"
        echo -e "     ${_BLUE}bash nebula_scripts/submit_job.sh nebula_scripts/grpo/grpo_ranking_gsl.sh 1 lazada_llm_ad_h20${_NC}"
    else
        echo ""
        echo -e "${_RED}⚠️  部分依赖缺失，请运行以下命令安装:${_NC}"
        echo -e "     ${_BLUE}pip install -r requirements_flex.txt --trusted-host mirrors.aliyun.com -i http://mirrors.aliyun.com/pypi/simple/${_NC}"
    fi
    echo ""
}

# =============================================================================
# 主流程
# =============================================================================

main() {
    echo ""
    echo "============================================================"
    echo "  TASD 环境初始化"
    echo "  项目路径: ${TASD_ROOT}"
    echo "============================================================"

    # 1. 检测环境
    local env_type
    env_type=$(detect_environment)
    echo -e "   环境类型: ${_GREEN}${env_type}${_NC}"

    # 2. 激活 Python 环境
    activate_python_env "${env_type}"

    # 3. 设置通用环境变量
    setup_common_env

    # 4. 验证环境
    verify_environment
}

# 执行主流程
main
