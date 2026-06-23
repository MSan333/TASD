# GRPO v7 快速参考

## 一行命令

```bash
# 本地单卡训练（快速验证）
bash exp/grpo/start/train_local.sh

# 提交星云 4 卡训练
bash exp/grpo/start/submit_nebula.sh

# 覆盖超参
bash exp/grpo/start/submit_nebula.sh LR=5e-6 TOTAL_TRAINING_STEPS=200

# 多节点（2 节点 × 4 卡 = 8 卡）
WORLD_SIZE=2 bash exp/grpo/start/submit_nebula.sh
```

## v7 核心改动

| 改动 | 说明 | 文件 |
|------|------|------|
| KL 正则化 | `use_kl_loss=True, kl_loss_coef=0.01` | `train_local.sh`, `grpo_ranking_gsl_v4.sh` |
| DAPO filter | 格式无效 → `score=0`（GRPO 自动产生负 advantage） | `reward_fn.py`, `ranking_batch.py` |
| Top-3 Precision | 只看前 3 位置，归一化 `[0,1]` | `reward_fn.py`, `ranking_batch.py` |
| Advantage 归一化 | `norm_adv_by_std_in_grpo=True` | `train_local.sh`, `grpo_ranking_gsl_v4.sh` |
| 去掉 rollout_is | 策略漂移后 IS 变成噪声 | `train_local.sh`, `grpo_ranking_gsl_v4.sh` |

## Reward 公式

```python
if format_valid:
    top3_score = top3_precision(rank_list)  # ∈ [0, 1]
    if has_judge:
        judge_norm = (judge_score + 1) / 2  # [-1,1] → [0,1]
        score = 0.7 * top3_score + 0.3 * judge_norm
    else:
        score = top3_score
else:
    score = 0.0  # DAPO filter
```

## 监控指标

| 指标 | 期望值 | 说明 |
|------|--------|------|
| `reward/rule_score/mean` | 逐步上升 | Top-3 Precision，核心指标 |
| `rollout_corr/kl` | < 0.1 | KL 散度，v4 爆炸到 0.7 |
| `actor/entropy` | 1-3 | 策略熵，v4 涨到 5.5 |
| `reward/format_valid_rate` | 前 20 步快速上升 | 格式合法率 |

## 文件位置

```
exp/grpo/
├── start/
│   ├── train_local.sh        # 本地训练
│   ├── submit_nebula.sh      # 星云提交（含密钥）
│   └── setup.sh              # 环境安装
├── b/
│   ├── NEBULA_TRAINING.md    # 详细指南
│   └── QUICK_REFERENCE.md    # 本文件
├── reward_fn.py              # v7 reward（单条）
└── ranking_batch.py          # v7 reward（batch）

nebula_scripts/grpo/
└── grpo_ranking_gsl_v4.sh    # 星云训练脚本
```

## 详细文档

→ [NEBULA_TRAINING.md](NEBULA_TRAINING.md)  
→ [v7 Reward 设计](../../docs/v7_reward_redesign.md)
