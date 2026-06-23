# v7 Reward 函数重设计说明

## 核心问题（v4 训练诊断）

v4 训练 84 步后模型输出变成乱码。深度诊断发现：

1. **KL 正则化完全未开启**
   - `use_kl_loss=False` + `use_kl_in_reward=False`
   - `actor/kl_loss=0.0`（全程）
   - KL 从 0.003 无约束爆炸到 0.7

2. **Format penalty 占 reward 的 64-94%**
   - Step 1: `score=-1.52`, `format_penalty=-1.40` (92%)
   - Step 50: `score=-0.58`, `format_penalty=-0.37` (64%)
   - Step 84: `score=-0.42`, `format_penalty=-0.31` (74%)
   - 模型只学会了"输出合法 JSON"，排序质量 84 步没有提升

3. **策略漂移**
   - Entropy 从 0.5 涨到 5.5（趋于随机）
   - Rollout IS ratio 从 1.0 降到 0.2（修正信号变成噪声）
   - 最终输出变成多语言乱码

## v7 设计（基于论文）

### 论文依据

| 论文 | 核心技术 | 对本任务的启发 |
|------|---------|--------------|
| **DAPO** (ByteDance 2025) | Overlong filtering: 超长输出不参与梯度计算 | Format 无效样本应该被 **filter** 而非 **penalize** |
| **DeepSeek-R1** (2025) | reward = accuracy_reward + format_reward，两个正交分量，都在 [0,1] | Reward 分量应该**正交、同范围、可解释** |
| **Kimi-k1.5** (2025) | Length penalty 作为 soft constraint + 高质量验证 | 约束和质量分离，质量信号要足够强 |
| **PRM** (Lightman et al., 2023) | Process Reward: 每步都有奖励信号，而非只在终点 | 更密集的奖励信号 → 更快收敛 |
| **Math-Shepherd** (2024) | Step-level verifier + 逐步验证 | 位置级别的排序质量评分 |

### v7 Reward 公式

```python
# DAPO-style filter + Top-3 Precision

if format_valid:
    top3_score = top3_precision(rank_list, positive_keys, negative_keys)
    # top3_score ∈ [0, 1]，只看前 3 个位置
    
    if has_judge:
        judge_norm = (judge_score + 1.0) / 2.0  # [-1, 1] → [0, 1]
        final_score = 0.7 * top3_score + 0.3 * judge_norm
    else:
        final_score = top3_score
else:
    final_score = 0.0  # DAPO filter: GRPO 自动产生负 advantage
```

### Top-3 Precision 评分

```python
_TOP3_WEIGHTS = {0: 1.0, 1: 0.5, 2: 0.33}

def top3_precision(rank_list, positive_keys, negative_keys):
    # 正样本在 Top-3: 加权奖励 (pos0=1.0, pos1=0.5, pos2=0.33)
    # 正样本不在 Top-3: 0 (业务不曝光，无所谓)
    # 负样本在 Top-3: 加权惩罚
    # 负样本不在 Top-3: 0 (业务不曝光，无所谓)
    
    pos_score = 0.0
    max_pos = 0.0
    for key in positive_keys:
        if key in rank_list:
            pos = rank_list.index(key)
            if pos < 3:
                w = _TOP3_WEIGHTS[pos]
                pos_score += w
                max_pos += w
    
    neg_score = 0.0
    max_neg = 0.0
    for key in negative_keys:
        if key in rank_list:
            pos = rank_list.index(key)
            if pos < 3:
                w = _TOP3_WEIGHTS[pos]
                neg_score -= w
                max_neg += w
    
    # 归一化到 [0, 1]
    pos_norm = (pos_score + max_pos) / (2 * max_pos) if max_pos > 0 else 0.5
    neg_norm = (neg_score + max_neg) / (2 * max_neg) if max_neg > 0 else 0.5
    
    return 0.6 * pos_norm + 0.4 * neg_norm
```

### 为什么这个设计更好？

| 维度 | v4 (旧) | v7 (新) | 改进 |
|------|---------|---------|------|
| **Format 处理** | 加法惩罚 (`score += format_penalty`) | DAPO filter (`invalid=0`) | 约束与质量解耦，避免 format 主导 reward |
| **Reward 范围** | `[-2.0, 0.35]`（format 和 quality 混在一起） | `[0, 1]`（纯质量） | 同范围，GRPO 方差更稳定 |
| **位置关注** | 所有位置都有信号 | 只看 Top-3 | 符合业务需求（只曝光 Top-3） |
| **组内方差** | 小（format_penalty 全 ≈ -0.2） | 大（不同排列差异明显） | GRPO 梯度信号更强 |
| **Judge 归一化** | 未归一化（`[-1, 1]`） | 归一化到 `[0, 1]` | 与 top3 同范围，可解释 |

## 关键改动文件

1. **`exp/grpo/reward_fn.py`**
   - `_rule_based_score()` → Top-3 Precision（只看前 3 个位置）
   - `compute_reward()` → DAPO filter + judge 归一化

2. **`exp/grpo/ranking_batch.py`**
   - `_batch_with_llm()` → 同步 v7 公式
   - `_batch_no_llm()` → 简化实现

3. **`exp/grpo/start/train_local.sh`**
   - `use_kl_loss=True`, `kl_loss_coef=0.01`
   - `norm_adv_by_std_in_grpo=True`
   - 去掉 `rollout_correction.rollout_is`

4. **`nebula_scripts/grpo/grpo_ranking_gsl_v4.sh`**
   - 同步所有 v7 修改

## 运行验证

```bash
# 本地单卡测试
bash exp/grpo/start/train_local.sh

# 观察指标
# 1. rollout_corr/kl 保持 < 0.1（v4 爆炸到 0.7）
# 2. pure_ranking_score = 0.7*rule + 0.3*judge_norm 应该逐步提升
# 3. entropy 保持在 1-3（v4 涨到 5.5）
# 4. 模型输出保持为有效 JSON（v4 后期变乱码）
```

## 预期效果

1. **Format 学习更快**
   - v4: 50 步才稳定到 90% format_valid
   - v7: 预计 10-20 步（DAPO filter 提供更清晰的信号）

2. **排序质量真正提升**
   - v4: 84 步 pure_ranking_score 基本不变
   - v7: 预计 50 步内看到明显提升

3. **训练更稳定**
   - v4: KL 爆炸到 0.7，策略崩溃
   - v7: KL 保持在 0.1 以下（use_kl_loss=True 生效）

4. **训练更快**
   - v4: 每步 ~350 秒
   - v7: 预计 ~280 秒（format 无效样本不计算 reward）

## 参考论文

1. **DAPO**: Decoupled Clip and Dynamic Overlong Penalty for LLM Alignment
   - ByteDance, 2025
   - 提出 overlong filtering 技术

2. **DeepSeek-R1**: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
   - DeepSeek, 2025
   - Reward 分量正交、同范围

3. **PRM**: Process Reward Models for LLMs
   - Lightman et al., 2023
   - 过程奖励信号

4. **Math-Shepherd**: Step-level Supervision for Mathematical Reasoning
   - 2024
   - 逐步验证技术

5. **GRPO**: Group Relative Policy Optimization
   - DeepSeek, 2024
   - 组内方差计算 advantage
