# GRPO Ranking Reward 函数迭代记录

> 记录每个版本的训练配置、结果和教训，用于避免重复踩坑。
> 
> 基座模型: Qwen3-8B | 数据集: minirank | 框架: veRL (GRPO)
> 
> 最后更新: 2026-06-29

---

## 目录

| 版本 | 日期 | Reward 公式 | 核心问题 | 状态 |
|------|------|------------|---------|------|
| [v2-v3](#v2-v3-初始版本) | 06-09 | 三层混合 (格式+eval+rule) | 初始搭建, 未系统训练 | 📦 基线 |
| [v4](#v4-星云首训) | 06-23~24 | 0.6×rule + 0.4×judge | KL 爆炸, 84 步崩溃 | ❌ 失败 |
| [v5/v5.1](#v5v51-混合-reward--entropy-boost) | 06-24~25 | 0.5×rule + 0.5×judge + entropy boost | rule 被钻空子, judge 无信号 | ❌ 失败 |
| [v7](#v7-dapo-filter--top-3-precision) | 06-25 | 0.7×top3 + 0.3×judge_norm + DAPO filter | 首次有效框架, 但 judge 权重太低 | ⚠️ 部分有效 |
| [v8](#v8-混合公式调整) | 06-25 | 0.3×rule + 0.7×judge + format_penalty | rule 被钻空子, markdown 无效 | ❌ 失败 |
| [v9](#v9-pure-judge--format-penalty) | 06-26 | judge_norm + format_penalty | 信号倒挂, invalid>valid | ❌ 失败 |
| [v10](#v10-progressive-invalid--clean-bonus) | 06-28 | judge_norm + progressive_invalid + clean_bonus | judge 无区分度, entropy 坍缩 | ⚠️ 格式OK但学不动 |
| [v11](#v11-pairwise-preference-reward) | 06-29 | rule_base + pairwise_preference + clean_bonus | 训练中 | 🔄 进行中 |

---

## v2-v3: 初始版本

**日期**: 2026-06-09  
**Commit**: `4a20480` (v2), `d152fc8` (v3)

### 配置

```
Reward: 三层混合架构
  L0: 格式门槛 (JSON 数组, ≥3 张有效卡片)
  L1: eval_score (LLM Judge)
  L2: rule_score (标签位置评分)
  
公式: eval_weight × eval_score + rule_weight × rule_score + format_penalty
```

### 改动

- v2: 基础 reward 框架 + LLM Judge batch 评分
- v3: 新增 Knowledge Judge (TipBank 80 skills + 意图识别), SwanLab 监控

### 教训

- 未完成系统训练, 仅作为框架搭建
- v3 引入了 Knowledge Judge 但缺少数据验证

---

## v4: 星云首训

**日期**: 2026-06-23 ~ 06-24  
**Commit**: `cf67f30` (事后回溯)  
**运行数**: 5+ (outputs/2026-06-23, 2026-06-24)

### 配置

```yaml
# 训练超参
LR: 1e-5
train_batch_size: 8
mini_batch_size: 2
rollout_n: 4
entropy_coeff: 0.02          # 有 entropy 正则
kl_loss_coef: 0.003          # KL loss (但未真正生效, 见下)
kl_coef: 0.001               # KL reward penalty
norm_adv_by_std: true

# Reward
公式: 0.6 × rule_score + 0.4 × judge_score + format_penalty
format_penalty: -0.5 (加法惩罚, 格式无效时)
```

### 训练结果 (84 步后崩溃)

```
Step 1:   score=-1.52, format_penalty=-1.40 (92%), KL=0.003
Step 50:  score=-0.58, format_penalty=-0.37 (64%), KL=0.007
Step 84:  score=-0.42, format_penalty=-0.31 (74%), KL=0.7 ❌ 爆炸
         → 输出变成多语言乱码, 策略完全漂移
```

### 根因分析

1. **`use_kl_loss=False`**: KL 正则化完全未开启 (actor/kl_loss=0 全程), 只靠 reward 中的 kl_penalty 不够
2. **格式惩罚用加法**: 格式无效时 `score -= 0.5`, 导致 reward 全负, GRPO 无法区分好坏
3. **`entropy_coeff=0.02` 在 KL 未开启时无效**: 策略自由漂移, entropy 从 2.0 涨到 5.5

### 教训

> ❗ **KL loss 必须开启** (`use_kl_loss=True, kl_loss_coef=0.01`), 仅靠 reward 中的 KL penalty 不够
> 
> ❗ **格式惩罚不能用简单加法**: 全负 reward 让 GRPO 的 advantage 计算失效
> 
> ❗ **entropy_coeff 需配合 KL loss 使用**: 没有 KL 锚定时 entropy 正则化形同虚设

---

## v5/v5.1: 混合 Reward + Entropy Boost

**日期**: 2026-06-24 ~ 06-25  
**Commit**: `7cba990` (v5), `49a5c72` (v5.1)

### 配置

```yaml
# Reward
公式: 0.5 × rule_score + 0.5 × judge_score + format_penalty
rule_score: _rule_based_score (Top-3 标签评分 + 规则合规性)
judge_model: deepseek-v4-flash

# 超参
kl_loss_coef: 0.01           # 修复: 开启 KL loss
entropy_coeff: 0.02          # 保持 entropy boost
kl_coef: 0.15                # v5: 加强 KL (过强, v5.1 改回)
```

### v5.1 修复

- `_detect_format_issues` 返回 list 而非 dict 的 bug
- 格式无效输出仍获得正 rule_score (退化为 "输出 1 个正样本 + 垃圾 = +0.20")
- checkpoint save deadlock (rank 0 先删再 barrier → timeout)
- 添加小型验证集 (100 prompts vs 1982) 加速验证

### 训练结果

```
rule_score: 稳步上升 (有效)
judge_score: 全程 ≈ -0.2 (Judge 无信号)
问题: 模型学会了 "钻规则空子" — rule_score 涨但排序质量不变
```

### 根因分析

1. **rule_score 可被博弈**: 模型发现输出特定组合就能拿高分, 不需要真正理解排序逻辑
2. **judge_score 全程无信号**: Judge 模型 (deepseek-v4-flash) 打分集中在 [-0.4, -0.1], 无区分度
3. **50/50 权重不合理**: rule 可博弈 + judge 无信号 = 两个信号都不可靠

### 教训

> ❗ **rule_score 单独作为 reward 信号会被博弈**: 模型会找捷径拿分而非学习排序质量
> 
> ❗ **judge 权重不能太低**: 50/50 时 rule 的博弈空间主导了学习方向
> 
> ✅ **KL loss 开启后稳定性大幅提升**: v5.1 不再出现 v4 的崩溃

---

## v7: DAPO Filter + Top-3 Precision

**日期**: 2026-06-25  
**Commit**: `cf67f30`

### 配置

```yaml
# Reward (基于 DAPO/DeepSeek-R1/PRM 论文)
公式: 0.7 × top3_precision + 0.3 × judge_norm
top3_precision: 只看前 3 位置, 归一化 [0,1]
judge_norm: (judge_score + 1) / 2 ∈ [0, 1]

# 格式
DAPO filter: 格式无效 → score=0 (GRPO 自动产生负 advantage)

# 超参
kl_loss_coef: 0.01
entropy_coeff: 0
norm_adv_by_std_in_grpo: true
rollout_is: 去掉 (策略漂移后 IS 变成噪声)
```

### 设计改进

1. **DAPO filter**: 格式无效直接 score=0, GRPO 自然给负 advantage (论文: ByteDance DAPO 2025)
2. **Top-3 Precision**: 业务只曝光 Top-3, 只看前 3 位置评分
3. **位置权重**: pos0→1.0, pos1→0.5, pos2→0.33, pos3+→0

### 训练结果

```
框架首次有效: GRPO 能区分好坏, reward 有上升趋势
问题: judge 权重只有 0.3, rule 信号仍占主导 → 仍可能被博弈
```

### 教训

> ✅ **DAPO filter 方向正确**: 格式无效 → 0 比加法惩罚更简洁
> 
> ✅ **Top-3 Precision 符合业务逻辑**: 比全位置评分更有意义
> 
> ⚠️ **judge 权重 0.3 可能太低**: rule 仍占 0.7, 博弈风险未消除

---

## v8: 混合公式调整

**日期**: 2026-06-25  
**运行数**: 8 (outputs/2026-06-25, 14:37 ~ 18:46)

### 配置

```yaml
# Reward
公式: 0.3 × rule_score + 0.7 × judge_score + format_penalty
format_penalty: markdown -0.1, 幻觉 -0.15/key

# 超参
LR: 1e-5
train_batch_size: 4
mini_batch_size: 2
rollout_n: 4
kl_loss_coef: 0.01
entropy_coeff: 0
```

### 训练结果

```
rule_score: 0.08 → 0.75 (被钻空子, 同 v5)
judge_score: 全程 ≈ -0.2 (Judge 无区分度)
has_markdown: 0.5 → 1.0 (markdown 惩罚 -0.1 完全无效, 模型全输出 markdown)
KL: 0.005 → 0.090 (漂移但可控)
```

### 根因分析

1. **rule_score 仍被博弈**: 虽然降到 0.3 权重, 但模型仍优先优化 rule
2. **markdown 惩罚 -0.1 太弱**: 模型发现 markdown 输出更容易解析, 惩罚不痛不痒
3. **batch/single 代码不一致**: `_batch_with_llm` 仍用 v8 公式但 `compute_reward` 已更新 → 训练时走 batch, 配置写的是另一个公式

### 教训

> ❗ **惩罚值必须足够大才有行为约束力**: -0.1 对模型完全无感
> 
> ❗ **batch 和 single reward 代码必须统一**: 代码不一致导致训练行为不可预测
> 
> ❗ **judge 无区分度是跨版本的核心问题**: v4/v5/v7/v8 都出现 judge ≈ 常数

---

## v9: Pure Judge + Format Penalty

**日期**: 2026-06-26  
**运行数**: 2 (outputs/2026-06-26, 15:35, 17:24)

### 配置

```yaml
# Reward (去掉 rule_score, 消除博弈)
公式: judge_norm + format_penalty
judge_norm: (judge_score + 1) / 2 ∈ [0, 1]
format_penalty: markdown -0.3, 幻觉 -0.15/key

# 格式
DAPO filter: 格式无效 → score=0

# 超参 (同 v8)
LR: 1e-5, batch=4, rollout=4, kl_loss=0.01, entropy=0
```

### 训练结果 (125 步)

```
avg_score: -0.57 (100% 负值!)
avg_judge: -0.02 (Judge 无区分度)
format_valid: 很低, 大量输出被判为非法
pg_loss: ≈ 0 (策略几乎不动)
has_markdown: 高 (markdown 惩罚 -0.3 反而把有效输出拖到 -0.6)

核心问题:
  invalid 输出 score=0 (DAPO filter)
  valid 输出 score≈-0.57 (judge_norm≈0.5 + penalty -0.3 ≈ 0.2... 但实际更低)
  → invalid (0) > valid (-0.57) ❌ 信号倒挂!
  → GRPO 给非法输出正 advantage, 鼓励输出非法格式
```

### 根因分析

1. **信号倒挂**: DAPO filter (invalid=0) 高于 valid 输出的平均分 (-0.57), GRPO 反向学习
2. **markdown 惩罚 -0.3 过激**: 把本来 judge_norm ≈ 0.5 的有效输出拖到 0.2 甚至更低
3. **Judge 无区分度**: judge_score 集中在 [-0.3, +0.3], 归一化后 [0.35, 0.65], 差距太小
4. **batch 仍在用 v8 公式**: 代码不一致问题未修复

### 教训

> ❗ **DAPO filter (invalid=0) 必须确保 valid_avg > 0**: 否则 GRPO 鼓励非法输出
> 
> ❗ **惩罚叠加不能把有效输出拖到负值**: markdown -0.3 + 其他惩罚可能让 valid < invalid
> 
> ❗ **Judge 打分需要校准**: 直接给 [-1, +1] 原始分不够, 需要明确引导正分范围
> 
> ❗ **代码不一致是致命 bug**: batch/single 必须走同一个函数

---

## v10: Progressive Invalid + Clean Bonus

**日期**: 2026-06-28  
**运行数**: 2 (outputs/2026-06-28, 16:34, 16:48)  
**运行时长**: ~2h, 19 steps (未完成 500 steps)

### 配置

```yaml
# Reward (v10 修复信号倒挂)
公式: judge_norm + format_penalty + clean_bonus
judge_norm: (judge_score + 1) / 2 ∈ [0, 1]
format_penalty: markdown -0.1 (降低), extra_text -0.05 (新增)
clean_bonus: 纯 JSON 输出 +0.1
progressive_invalid: 
  部分合法: max(0, 0.25 - 0.1 × (3 - n_valid))
  完全非法: -0.3

# 超参
LR: 1e-5
train_batch_size: 4
mini_batch_size: 2
rollout_n: 4
kl_loss_coef: 0.01
entropy_coeff: 0
max_prompt_length: 6144
max_response_length: 1024
```

### 训练结果 (19 steps)

#### 格式合法率 — ✅ 大幅改善

```
Step 0:  19% (3/16)
Step 7:  81% (13/16)
Step 12: 100% (16/16)
Val@10:  88.7% (323/364)
Val@20:  95.3% (347/364)
```

Progressive invalid scoring 成功解决信号倒挂。Clean bonus 生效 — step 13 起所有 valid 输出的 `avg_clean=0.1000`。

#### 平均 Score — ⚠️ 有改善但幅度有限

```
Step 0-3:  -0.19 ~ -0.13  (大部分 invalid 拉低)
Step 4-8:  -0.05 ~ +0.09  (格式改善后转正)
Step 9-19: +0.08 ~ +0.51  (波动大, 均值 ~0.25)
```

#### Entropy — ❌ 严重坍缩

```
Step 1:  0.465
Step 7:  0.258
Step 13: 0.111
Step 17: 0.068
Step 19: 0.074
→ 85% 坍缩, 输出高度确定, 丧失探索能力
```

#### pg_loss — ❌ 接近零

整段 `pg_loss ∈ [-0.005, +0.005]`, 策略几乎不更新。

#### KL — ✅ 稳定

`rollout_corr/kl ∈ [0.001, 0.006]`, 无发散。

#### Judge — ❌ 核心瓶颈

```
avg_llm (Judge 原始分): -0.14 ~ -0.74, 中位数 ~ -0.55
judge_norm 集中在 [0.13, 0.23] — 窄带无区分度
rule_score: 固定在 ~0.7 (模型找到 "安全" 排序模式)

单条 score 示例:
  {'score': 0.225, 'rule_score': 0.7, 'judge_score': -0.7}
  {'score': 0.25,  'rule_score': 0.7, 'judge_score': -0.7}
```

### 根因分析

1. **Judge 打分过于集中在负值区**: 即使是 100% 合法格式, Judge 也给 -0.5~-0.7
   - judge_norm 集中在 [0.15, 0.25], GRPO 组内差异 ≈ 0 → advantage ≈ 0
   - 模型只学会了 "输出合法 JSON + 拿 clean bonus", 排序质量未改善
2. **entropy_coeff=0 导致坍缩**: 无探索正则, 模型输出趋于确定
3. **batch_size=4 + rollout_n=4**: GRPO 组内对比样本太少, advantage 噪声大
4. **Judge 校准 prompt 不够明确**: 需要告诉 Judge "合理排序应给正分"

### 教训

> ✅ **Progressive invalid scoring 有效**: 格式合法率从 19% → 100%, 信号方向正确
> 
> ✅ **Clean bonus 有效**: 模型学会输出纯 JSON
> 
> ✅ **降低 markdown 惩罚 (-0.3 → -0.1) 正确**: 不再把有效输出拖到负值
> 
> ❗ **Judge 无区分度是跨版本的核心瓶颈**: v4/v5/v7/v8/v9/v10 全都遇到
>   - 根因可能是: (a) Judge prompt 校准不够 (b) Judge 模型偏严 (c) 排序任务本身难以用 LLM 打分
> 
> ❗ **entropy_coeff=0 在单卡小 batch 下危险**: rollout_n=4 已经样本少, 再加无探索 → 快速坍缩
> 
> ❗ **纯 Judge reward 不可行**: Judge 信号太弱, 无法驱动策略更新

---

## v11: Pairwise Preference Reward

**日期**: 2026-06-29  
**状态**: 训练中

### 配置

```yaml
# Reward (Pairwise Preference)
公式: base_score + α × judge_preference + format_penalty + clean_bonus
base_score: rule_score (确定性, 无 LLM)
judge_preference: pairwise 比较 {-1, 0, +1}
α: 0.3 (pairwise 权重)
format_penalty: markdown -0.1, extra_text -0.05
clean_bonus: 纯 JSON +0.1
progressive_invalid: 部分合法 max(0, 0.25 - 0.1 × (3 - n_valid)), 完全非法 -0.3

# 超参 (vs v10 改动)
entropy_coeff: 0.02          # v10 是 0 (导致 entropy 坍缩)
rollout_n: 8                 # v10 是 4 (更多 rollout, pairwise 比较更充分)
# 其余同 v10: LR=1e-5, kl_loss=0.01, batch=4, mini=2
```

### 设计改进

1. **Pairwise Preference Reward**: 相对比较替代绝对打分
   - 组内 pairwise 比较: 每个 rollout vs 中位数锚点
   - Judge 输出离散 preference {-1, 0, +1}, 强制有方差
   - 解决 v10 Judge 无区分度问题 (judge_norm 集中在 [0.15, 0.25])
2. **entropy_coeff: 0 → 0.02**: 防止 entropy 坍缩 (v10: 0.465 → 0.074)
3. **rollout_n: 4 → 8**: 更多 rollout, pairwise 比较更充分
4. **保留 v10 有效做法**: progressive invalid + clean bonus

### 训练结果

待观察 (训练中)

### 预期效果

```
judge_preference 分布: {-1: ~33%, 0: ~33%, +1: ~33%} (均匀分布)
entropy: 稳定在 [0.3, 0.5] (不坍缩)
pg_loss: >0.01 (有非零梯度)
base_score: 逐步上升 (排序质量改善)
```

### 关键文件

- `exp/grpo/knowledge_judge.py`: 新增 `async_pairwise_compare`, `pairwise_compare`, `parse_pairwise_response`
- `exp/grpo/ranking_batch.py`: 重写 `_batch_with_llm` 使用 pairwise 方案
- `exp/grpo/start/train_local_v11.sh`: v11 训练脚本

---

## 跨版本总结

### 已确认的有效做法

| 做法 | 首次引入 | 验证版本 |
|------|---------|---------|
| `use_kl_loss=True, kl_loss_coef=0.01` | v5 | v7/v8/v9/v10 ✅ |
| DAPO filter / Progressive invalid | v7/v10 | v10 ✅ (格式合法率 100%) |
| Top-3 Precision 位置权重 | v7 | v7+ ✅ (符合业务逻辑) |
| `norm_adv_by_std_in_grpo=True` | v7 | v7+ ✅ |
| Clean bonus (+0.1 纯 JSON) | v10 | v10 ✅ |
| batch/single 代码统一 | v10 | v10 ✅ (消除不一致 bug) |

### 已确认的失败模式

| 模式 | 出现版本 | 根因 |
|------|---------|------|
| KL 爆炸 → 策略漂移 | v4 | `use_kl_loss=False` |
| rule_score 被博弈 | v5/v8 | 确定性规则可被模型找到捷径 |
| Judge 无区分度 | v4~v10 全部 | Judge prompt 校准 / 模型偏严 |
| 信号倒挂 (invalid > valid) | v9 | DAPO filter=0 + 惩罚过重 |
| Entropy 坍缩 | v10 | entropy_coeff=0 + 小 batch |
| pg_loss ≈ 0 | v9/v10 | Judge 窄带 → advantage ≈ 0 |

### 下一步方向

1. **解决 Judge 区分度问题** (最优先):
   - 方案 A: 改善 Judge prompt 校准 (明确正分范围)
   - 方案 B: 换更强的 Judge 模型
   - 方案 C: 用 preference-based reward 替代 absolute score (对比排序而非绝对打分)
   
2. **entropy_coeff > 0**: 至少 0.01, 防止坍缩

3. **增大 rollout_n**: 4 → 8, 让 GRPO 组内对比更充分

4. **考虑不依赖 LLM Judge 的 reward**:
   - 纯 rule_score (需要更精细、不可博弈的规则)
   - Preference ranking (给两个排序让 Judge 选更好的)
   - Self-play reward (模型自己评估排序质量)

---

## 附录: 公共配置

所有版本共享的基础配置:

```yaml
model: Qwen3-8B
dataset: minirank (train ~5400 条, val ~100 条)
max_prompt_length: 6144
max_response_length: 1024
max_model_len: 8192
optimizer: AdamW (lr_warmup_steps=10)
FSDP: bfloat16, reshard_after_forward=true
rollout: vLLM (temperature=1.0, top_p=0.9, top_k=-1)
logger: [console, swanlab]
save_freq: 10
test_freq: 10
```
