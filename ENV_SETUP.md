# TASD 环境配置与运行指南

本文档覆盖 **三种运行场景** 的完整配置流程：新镜像（星云/Docker）、老镜像（星云/Docker）、Notebook 本地开发。

---

## 镜像信息

| 镜像 | 地址 | 说明 |
|------|------|------|
| **新镜像** | `hub.docker.alibaba-inc.com/mdl/notebook_saved:guoshuaile.gsl_33v2_20260609193943` | 最新环境，所有依赖已预装 |
| **老镜像** | `hub.docker.alibaba-inc.com/mdl/notebook_saved:guoshuaile.gsl_llm_rl_20260605230018` | 基础环境，可能需要增量安装部分依赖 |

---

## 场景一：新镜像（推荐）

新镜像已预装完整的训练环境，开箱即用。

### 1.1 星云提交

```bash
cd /path/to/TASD

# 加载环境变量
source exp/grpo/env.sh

# 使用新镜像提交（自动使用 setup_env.sh 中的默认新镜像）
CUSTOM_DOCKER_IMAGE="hub.docker.alibaba-inc.com/mdl/notebook_saved:guoshuaile.gsl_33v2_20260609193943" \
bash nebula_scripts/submit_job.sh \
    nebula_scripts/grpo/grpo_ranking_gsl.sh \
    1 \
    lazada_llm_ad_h20
```

> 提交前确保已设置 `OPENLM_TOKEN`、`OSS_ACCESS_ID`、`OSS_ACCESS_KEY`（详见[账号配置](#账号配置)）。

### 1.2 Docker 本地运行

```bash
cd /path/to/TASD

# 拉取新镜像
docker pull hub.docker.alibaba-inc.com/mdl/notebook_saved:guoshuaile.gsl_33v2_20260609193943

# 启动容器（挂载代码和模型目录）
docker run --gpus all -it --ipc=host --shm-size=16g \
    -v $(pwd):/workspace/TASD \
    -v /home/guoshuaile.gsl/models:/models \
    --name tasd_train \
    hub.docker.alibaba-inc.com/mdl/notebook_saved:guoshuaile.gsl_33v2_20260609193943 \
    bash

# 容器内操作
cd /workspace/TASD
source setup_env.sh            # 自动检测环境并初始化
bash run_local_grpo_ranking.sh  # 启动训练
```

---

## 场景二：老镜像

老镜像具备基础环境，需增量安装 TASD 特有依赖。

### 2.1 星云提交

```bash
cd /path/to/TASD
source exp/grpo/env.sh

CUSTOM_DOCKER_IMAGE="hub.docker.alibaba-inc.com/mdl/notebook_saved:guoshuaile.gsl_llm_rl_20260605230018" \
bash nebula_scripts/submit_job.sh \
    nebula_scripts/grpo/grpo_ranking_gsl.sh \
    1 \
    lazada_llm_ad_h20
```

> 星云提交时会自动通过 `requirements_nebula.txt` 安装缺失依赖。

### 2.2 Docker 本地运行

```bash
cd /path/to/TASD

docker pull hub.docker.alibaba-inc.com/mdl/notebook_saved:guoshuaile.gsl_llm_rl_20260605230018

docker run --gpus all -it --ipc=host --shm-size=16g \
    -v $(pwd):/workspace/TASD \
    -v /home/guoshuaile.gsl/models:/models \
    --name tasd_train_old \
    hub.docker.alibaba-inc.com/mdl/notebook_saved:guoshuaile.gsl_llm_rl_20260605230018 \
    bash

# 容器内操作
cd /workspace/TASD
source setup_env.sh  # 自动检测为老镜像，执行增量安装
bash run_local_grpo_ranking.sh
```

### 老镜像手动增量安装（如果 setup_env.sh 自动安装失败）

```bash
# 安装阿里源加速
pip install -r requirements_flex.txt \
    --trusted-host mirrors.aliyun.com \
    -i http://mirrors.aliyun.com/pypi/simple/

# 安装 TASD 项目
pip install -e . --no-deps --no-build-isolation

# 安装 Flash Attention（编译较慢，约 10-20 分钟）
export MAX_JOBS=4
FLASH_ATTENTION_FORCE_BUILD=TRUE pip install flash-attn --no-build-isolation \
    --trusted-host mirrors.aliyun.com \
    -i http://mirrors.aliyun.com/pypi/simple/
```

---

## 场景三：Notebook 本地开发

适用于开发机（如 Notebook 实例、工作站），无需 Docker。

### 3.1 环境初始化

```bash
cd /path/to/TASD

# 方式 A：使用 setup_env.sh 自动配置（推荐）
source setup_env.sh

# 方式 B：手动激活 conda 环境
conda activate python3.10
# 或使用 sdpo_env
# conda activate sdpo_env
```

### 3.2 首次安装依赖

```bash
# 安装 PyTorch（如未预装）
pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu124

# 安装项目依赖
pip install -r requirements_flex.txt \
    --trusted-host mirrors.aliyun.com \
    -i http://mirrors.aliyun.com/pypi/simple/

# 安装 TASD 项目（editable mode）
pip install -e . --no-deps --no-build-isolation

# 安装 Flash Attention
export MAX_JOBS=4
FLASH_ATTENTION_FORCE_BUILD=TRUE pip install flash-attn --no-build-isolation \
    --trusted-host mirrors.aliyun.com \
    -i http://mirrors.aliyun.com/pypi/simple/
```

### 3.3 启动训练

```bash
cd /path/to/TASD

# 加载环境变量
source exp/grpo/env.sh

# 启动训练（自动适配 GPU 数量）
bash run_local_grpo_ranking.sh

# 或指定 GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_local_grpo_ranking.sh

# 后台运行
nohup bash run_local_grpo_ranking.sh > outputs/training.log 2>&1 &
tail -f outputs/training.log
```

### 3.4 覆盖超参

```bash
# 改学习率
LR=5e-6 bash run_local_grpo_ranking.sh

# 改多个超参
LR=5e-6 TRAIN_BATCH_SIZE=16 KL_COEF=0.01 bash run_local_grpo_ranking.sh

# 改模型路径
MODEL_PATH=/path/to/your/model bash run_local_grpo_ranking.sh
```

---

## 数据准备

训练前需要确保数据和模型已就位。

### 训练数据

```bash
# 方式 1：自动下载
python -m exp.grpo.download_data

# 方式 2：从 OSS 手动下载
mkdir -p exp/grpo/data/ranking
ossutil64 cp oss://lazada-ai-model/ad/guoshauile.gsl/data/ranking/train.parquet exp/grpo/data/ranking/
ossutil64 cp oss://lazada-ai-model/ad/guoshauile.gsl/data/ranking/test.parquet exp/grpo/data/ranking/
```

### 模型权重

本地训练默认路径 `/home/guoshuaile.gsl/models/qwen3-8b`：

```bash
# 从 OSS 下载
mkdir -p ~/models/qwen3-8b
ossutil64 cp -r oss://lazada-ai-model/ad/guoshauile.gsl/model/qwen3-8b/ ~/models/qwen3-8b/
```

Docker 场景中已通过 `-v /home/guoshuaile.gsl/models:/models` 挂载，训练时设置：

```bash
MODEL_PATH=/models/qwen3-8b bash run_local_grpo_ranking.sh
```

---

## 账号配置

### 星云 / Nebula

提交星云任务需要以下环境变量（建议写入 `~/.bashrc`）：

```bash
export OPENLM_TOKEN="your_openlm_token"
export OSS_ACCESS_ID="your_oss_access_id"
export OSS_ACCESS_KEY="your_oss_access_key"
```

### DashScope（LLM Judge）

Reward 函数使用 DashScope API 打分，默认已在 `env.sh` 中配置：

```bash
export OPENAI_API_KEY="sk-93bf8a433943448bad6611ca5532a113"
export OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export OPENAI_MODEL="qwen3.6-max-preview"
```

如 API Key 失效，到 https://dashscope.console.aliyun.com 重新获取。

### SwanLab（日志追踪）

训练指标上传到 https://swanlab.cn ，默认已配置。如需更新：

```bash
export SWANLAB_API_KEY="your_swanlab_api_key"
```

---

## GPU 自适应配置

`run_local_grpo_ranking.sh` 会根据检测到的 GPU 数量自动调整超参：

| GPU 数量 | train_batch_size | mini_batch_size | rollout_n | val_n | 适用场景 |
|----------|-----------------|-----------------|-----------|-------|---------|
| 1 卡 | 8 | 2 | 4 | 4 | 本地调试 |
| 2 卡 | 16 | 4 | 4 | 8 | 小规模训练 |
| ≥ 4 卡 | 32 | 8 | 8 | 16 | 正式训练（与星云一致） |

---

## 常见问题

### Q: `ModuleNotFoundError: No module named 'ray'`

使用了错误的 Python 环境。运行 `source setup_env.sh` 自动检测并激活正确环境。

### Q: GPU 显存不足 (OOM)

降低 batch size：

```bash
MINI_BATCH_SIZE=1 TRAIN_BATCH_SIZE=4 ROLLOUT_N=2 GPU_MEM_UTIL=0.4 \
bash run_local_grpo_ranking.sh
```

### Q: Docker 容器中无法检测 GPU

确保使用 `--gpus all` 参数启动容器，并已安装 NVIDIA Container Toolkit：

```bash
# 验证 GPU 可见性
docker run --gpus all --rm nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

### Q: 老镜像中 `flash-attn` 安装失败

确保已安装 CUDA 开发工具包，并限制编译并发数：

```bash
export MAX_JOBS=2
FLASH_ATTENTION_FORCE_BUILD=TRUE pip install flash-attn --no-build-isolation
```

### Q: 星云提交报错 `OPENLM_TOKEN not set`

设置 Nebula 账号环境变量：

```bash
export OPENLM_TOKEN="your_token"
export OSS_ACCESS_ID="your_id"
export OSS_ACCESS_KEY="your_key"
```

### Q: 如何切换新/老镜像？

修改 `setup_env.sh` 中的 `DOCKER_IMAGE` 变量，或在星云提交时通过 `CUSTOM_DOCKER_IMAGE` 环境变量覆盖。

---

## 快速参考

### 一键命令速查

```bash
# ── 新镜像 + 星云提交 ──
source exp/grpo/env.sh
CUSTOM_DOCKER_IMAGE="hub.docker.alibaba-inc.com/mdl/notebook_saved:guoshuaile.gsl_33v2_20260609193943" \
bash nebula_scripts/submit_job.sh nebula_scripts/grpo/grpo_ranking_gsl.sh 1 lazada_llm_ad_h20

# ── 新镜像 + Docker 本地 ──
docker run --gpus all -it --ipc=host --shm-size=16g \
    -v $(pwd):/workspace/TASD -v ~/models:/models \
    hub.docker.alibaba-inc.com/mdl/notebook_saved:guoshuaile.gsl_33v2_20260609193943 bash
# 容器内: cd /workspace/TASD && source setup_env.sh && bash run_local_grpo_ranking.sh

# ── Notebook 本地 ──
source setup_env.sh && bash run_local_grpo_ranking.sh
```

### 关键文件

| 文件 | 作用 |
|------|------|
| `setup_env.sh` | 环境自动检测与初始化 |
| `exp/grpo/env.sh` | 训练超参与账号配置 |
| `run_local_grpo_ranking.sh` | 本地训练入口脚本 |
| `nebula_scripts/submit_job.sh` | 星云任务提交 |
| `requirements_flex.txt` | 弹性依赖（不锁版本） |
| `requirements_nebula.txt` | 星云增量依赖 |
