# 星云训练配置指南

## 环境变量配置

### 必须配置（星云提交需要）

提交训练任务前，必须在终端设置以下 3 个环境变量。建议写入 `~/.bashrc` 或 `~/.zshrc` 避免每次手动设置。

```bash
# ── 星云/Nebula 账号（必须）──
export OPENLM_TOKEN="你的openlm_token"
export OSS_ACCESS_ID="你的oss_access_id"
export OSS_ACCESS_KEY="你的oss_access_key"
```

**获取方式：**
- `OPENLM_TOKEN` — 从 OpenLM 平台获取，格式为 `OPENLM_xxxxx_`
- `OSS_ACCESS_ID` / `OSS_ACCESS_KEY` — 从阿里云 RAM 控制台获取，对应 OSS Bucket `lazada-ai-model` 的访问权限

### 可选配置

```bash
# ── LLM Judge API Key（Reward 打分用）──
# 获取地址：https://dashscope.console.aliyun.com
export OPENAI_API_KEY="你的dashscope_api_key"
```

如果不设置 `OPENAI_API_KEY`，Reward 函数中的 LLM Judge 将无法工作。

### 无需配置（已明文写入代码）

以下密钥已直接写入脚本中，无需手动设置：

| 变量 | 说明 |
|------|------|
| `SWANLAB_API_KEY` | SwanLab 日志追踪，已写入 `grpo_ranking_gsl.sh` 和 `launch_ray_cluster.sh` |

---

## 提交训练

### 1. 设置环境变量

```bash
# 复制并替换为你的真实密钥
export OPENLM_TOKEN="OPENLM_xxxxx_your_token"
export OSS_ACCESS_ID="LTAI5txxxxxxxxxx"
export OSS_ACCESS_KEY="your_oss_access_key"
export OPENAI_API_KEY="sk-xxxx"   # 可选
```

### 2. 提交任务

```bash
bash submit_grpo.sh
```

提交成功后会输出 Nebula logview 链接，可在浏览器中查看任务状态和日志。

### 3. 查看训练结果

| 内容 | 路径 |
|------|------|
| **Checkpoint** | `oss://lazada-ai-model/ad/guoshauile.gsl/result/<job_name>/` |
| **SwanLab 日志** | `oss://lazada-ai-model/ad/guoshauile.gsl/log/swanlab_logs/` |
| **SwanLab 网页** | https://swanlab.cn （项目名 `GRPO-Ranking`） |

---

## 训练超参

所有训练超参已硬编码在 `nebula_scripts/grpo/grpo_ranking_gsl.sh` 中，直接修改文件顶部数值即可：

| 参数 | 当前值 | 说明 |
|------|--------|------|
| `DATASET` | `ranking` | 数据集名称 |
| `LR` | `1e-5` | 学习率 |
| `TRAIN_BATCH_SIZE` | `32` | 训练 batch size |
| `MINI_BATCH_SIZE` | `8` | PPO mini batch |
| `ROLLOUT_N` | `8` | Rollout 采样数 |
| `KL_COEF` | `0.05` | KL 散度系数 |
| `TOTAL_TRAINING_STEPS` | `500` | 总训练步数 |
| `MODEL_PATH` | `/data/oss_bucket_0/ad/guoshauile.gsl/model/base/qwen3-8b` | 模型路径 |

---

## 常见问题

### Q: 提交时报 `OPENLM_TOKEN not set`

没有设置环境变量。执行：
```bash
export OPENLM_TOKEN="你的token"
```

### Q: SwanLab 日志看不到

确认 `SWANLAB_API_KEY` 是否正确。当前硬编码的 Key 如果失效，到 https://swanlab.cn/settings 重新获取。

### Q: LLM Judge 不工作

需要设置 `OPENAI_API_KEY` 环境变量：
```bash
export OPENAI_API_KEY="sk-xxxx"
```
