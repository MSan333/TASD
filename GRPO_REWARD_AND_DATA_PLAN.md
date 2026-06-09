# GRPO Reward Function 设计 + 在线采样数据构建方案

> 项目：rank-agent (Lazada 广告排序 Agent)
> 分支：exp/gsl_grpo
> 日期：2026-05-29
> 状态：代码已实现

---

## 一、背景

### 1.1 为什么直接用 GRPO，跳过 DPO

Prompt Engineering 优化已触达天花板（Pos Hit@1 ≈ 30%），需要通过模型微调突破。经过分析，**直接 GRPO 比先 DPO 再 GRPO 更适合当前场景**：

| DPO 的尴尬 | GRPO 的优势 |
|------------|------------|
| 数据天然不是 pair，硬凑 chosen/rejected 需要大量工程且引入额外噪声 | 不需要配对，采样 K 次 + reward 打分即可 |
| 只学 binary preference (A 比 B 好)，丢掉"好在哪里、好多少" | 多维 reward 精确告诉模型每个维度的得失 |
| 无法同时平衡 Pos Hit 和 Neg Avoid，两个指标可能此消彼长 | 多目标权重可调，同时优化多个指标 |
| 受限于已有数据，无法发现新的排序策略 | 在线探索 K=8 次采样，可能发现更好的排序模式 |
| 信号稀疏，pairwise loss 是二值信号 | NDCG 提供连续梯度信号 |

### 1.2 直接 GRPO 的前提验证

在正式训练前需要做一次**快速验证**：用目标模型对 20 个 sample 各采样 8 次，检查：

| 检查项 | 合格标准 | 不合格时的行动 |
|--------|---------|--------------|
| 格式合法率 | ≥ 50% (K=8 中至少 4 个合法输出) | 先做一轮 SFT warm start |
| 组内 reward 方差 | std > 0.1 | 增大 K 到 16 或提高 temperature |
| 最好 vs 最差 reward 差 | > 0.3 | 模型有区分能力，可以继续 |

如果 14B 模型 baseline 的 Pos Hit@1 > 20%，大概率直接 GRPO 即可。

### 1.3 整体路线

```
数据清洗 (dataset_cleaner)
  ↓ clean_pos / clean_neg 标签
在线采样 (grpo_data_builder)
  ↓ K=8 responses + rewards + advantages
GRPO 训练 (OpenRLHF / veRL)
  ↓ 微调后的 LoRA 权重
评测验收 (evaluator)
  ↓ Pos Hit@1 ≥ 38%
部署上线
```

---

## 二、Reward Function 设计

### 2.1 五层架构

```
输入: 模型原始输出 (raw text) + ground truth (clean_pos, clean_neg, card_pool)
  │
  ▼
┌─────────────────────────────────────────────────────────────────┐
│ L1: Format Gate (硬门槛)                                        │
│   - JSON 合法性、元素 ∈ card_pool、无重复、≥ 3 个 action        │
│   - 不过 → reward = -2.0, 直接返回                              │
└─────────────┬───────────────────────────────────────────────────┘
              │ 通过
              ▼
┌─────────────────────────────────────────────────────────────────┐
│ L2: Pos Hit Reward (权重 0.35)                                  │
│   - 正向动作 Hit@1 → +1.0                                      │
│   - 正向动作 Hit@3 → +0.5                                      │
│   - 未命中 → 0.0                                                │
│   - 多个正向动作取最高奖励                                       │
└─────────────┬───────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│ L3: Neg Avoid Reward (权重 0.30)                                │
│   - 负向动作 @Top-1 → -1.0                                     │
│   - 负向动作 @Top-3 → -0.5                                     │
│   - 全部拦截成功 → +0.3                                         │
│   - 多个负向动作取最严厉惩罚                                     │
└─────────────┬───────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│ L4: NDCG@5 (权重 0.20)                                         │
│   - 正向动作在 rank list 中的 NDCG                               │
│   - 连续信号 [0, 1]，提供梯度丰富的排序质量信息                   │
│   - 复用 dpo_scoring._compute_ndcg()                             │
└─────────────┬───────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│ L5: LLM Judge (权重 0.15, 可选)                                 │
│   - 灰色地带检测: 规则分在 [-0.3, 0.7] 时触发                    │
│   - 调 qwen-max API 做语义合理性评分                             │
│   - 返回 [-1, 1]                                                │
│   - 约 30% 样本触发, 单次 ~$0.01                                │
└─────────────┬───────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 加权汇总 + Clip                                                  │
│   total = 0.35*L2 + 0.30*L3 + 0.20*L4 + 0.15*L5                │
│   reward = clip(total, -2.0, +2.0)                              │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 权重设计逻辑

| 层 | 权重 | 设计理由 |
|----|------|---------|
| L2 Pos Hit | 0.35 | 最核心指标，正向动作排前面是首要目标 |
| L3 Neg Avoid | 0.30 | 第二重要，负向动作误排会直接伤害商家体验 |
| L4 NDCG | 0.20 | 连续信号补充 L2/L3 的稀疏性，提供更丰富的梯度 |
| L5 LLM Judge | 0.15 | 补充规则无法覆盖的语义层面判断，成本高所以权重低 |

L2 + L3 = 0.65 占主体，确保核心排序指标优先。

### 2.3 各层 Reward 值域

| 层 | 值域 | 极端好 | 极端差 |
|----|------|--------|--------|
| Format Gate | {0, -2} | 通过=0 | 非法=-2（直接截断） |
| Pos Hit | [0, 1] | Hit@1=1.0 | 全部未命中=0 |
| Neg Avoid | [-1, 0.3] | 全部拦截=+0.3 | Neg@Top-1=-1.0 |
| NDCG | [0, 1] | 完美排序=1.0 | 完全乱序=0 |
| LLM Judge | [-1, 1] | 非常合理=1.0 | 非常不合理=-1.0 |
| **总分** | **[-2, ~0.85]** | | |

### 2.4 LLM Judge 触发条件

```
规则型加权分 = 0.35 * L2 + 0.30 * L3 + 0.20 * L4

若 -0.3 ≤ 规则型加权分 ≤ 0.7 → 触发 LLM Judge
否则 → 不触发（规则已能给出明确判断）
```

### 2.5 验证结果

```
Good排序  [a,b,c,d,e,f] (pos=a,b neg=f): reward = 0.6400 ✅
中等排序  [c,a,d,b,e,f]               : reward = 0.3952 ✅
差排序    [f,e,d,c,b,a] (neg在Top-1)  : reward = -0.2526 ✅
格式非法  "this is not json"           : reward = -2.0000 ✅

单调性: Good > Medium > Bad > Invalid ✅
```

---

## 三、在线采样数据构建

### 3.1 采样流程

```
对每个 cleaned sample:
  │
  ├── 1. 构造 prompt (复用 dpo_data_builder._build_user_prompt)
  │
  ├── 2. API 采样 K=8 次 (temperature=0.8, 固定温度保证多样性)
  │       同步版: sample_k_responses()
  │       异步版: sample_k_responses_async()
  │
  ├── 3. 对 K 个响应逐个打分
  │       score_responses() → compute_grpo_reward() × K
  │
  ├── 4. 组内归一化
  │       advantages = normalize_advantages(rewards)
  │       A_i = (r_i - μ) / (σ + ε)
  │
  └── 5. 输出一条训练记录
```

### 3.2 组内归一化 (Advantage)

```python
A_i = (r_i - mean(rewards)) / (std(rewards) + eps)
```

**性质**：均值 ≈ 0，标准差 ≈ 1。不需要 Value Model，这就是 GRPO 的核心优势。

**验证**：
```
Rewards:    [0.64, 0.3952, -0.2526, -2.0, 0.64, 0.3952]
Advantages: [0.7208, 0.4576, -0.239, -2.1178, 0.7208, 0.4576]
Mean: 0.000000 ✅  Std: 1.0000 ✅
```

### 3.3 输出格式

```jsonl
{
  "prompt": "请帮我根据商家特征给出推荐的概率...",
  "system_prompt": "你是Lazada电商广告排序助手...",
  "responses": ["[a,b,c,d,e]", "[b,a,d,c,e]", ...],
  "rewards": [0.64, 0.40, -0.25, -2.0, 0.64, 0.40, 0.15, 0.48],
  "advantages": [0.72, 0.46, -0.24, -2.12, 0.72, 0.46, -0.12, 0.31],
  "reward_breakdowns": [
    {"total": 0.64, "format_valid": true, "pos_hit": 1.0, "neg_avoid": 0.3, "ndcg": 1.0, ...},
    ...
  ],
  "metadata": {"sample_id": "...", "venture": "VN", "k": 8, "temperature": 0.8, ...}
}
```

输出目录：`data/grpo/`

---

## 四、代码结构

```
grpo/                              # GRPO 专用模块
├── __init__.py
├── grpo_reward.py                 # 五层 Reward Function
│   ├── format_gate()              # L1
│   ├── pos_hit_reward()           # L2
│   ├── neg_avoid_reward()         # L3
│   ├── ndcg_reward()              # L4
│   ├── llm_judge_reward()         # L5
│   ├── should_trigger_llm_judge() # 灰色地带检测
│   └── compute_grpo_reward()      # 主入口
└── grpo_data_builder.py           # 在线采样数据构建
    ├── sample_k_responses()       # 同步采样
    ├── sample_k_responses_async() # 异步采样
    ├── score_responses()          # 批量打分
    ├── normalize_advantages()     # 组内归一化
    └── build_grpo_dataset()       # 主流程

dpo/                               # 被 GRPO 复用的基础设施
├── dpo_scoring.py                 # rank_of(), _compute_ndcg()
├── dpo_data_builder.py            # clean_sample_labels(), _build_user_prompt(), _parse_ranking()
└── ...
```

---

## 五、采样参数

| 参数 | 值 | 说明 |
|------|------|------|
| K | 8 | DeepSeek 验证过的平衡点 |
| temperature | 0.8 | 保证多样性，太低容易 collapse |
| min_clean_ratio | 0.6 | 标签置信度下限 |
| enable_llm_judge | False | 默认关闭，需要时手动开启 |
| clip_range | 2.0 | reward 裁剪范围 |

---

## 六、训练超参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| K | 8 | 采样数 |
| temperature | 0.8 | 采样温度 |
| clip_range (PPO) | 0.2 | 单步更新幅度 |
| kl_coef | 0.05 | KL 惩罚系数 |
| learning_rate | 5e-7 | LoRA 学习率 |
| batch_size | 4 prompts/GPU | |
| max_steps | 500 | 总训练步数 |
| lora_rank | 64 | LoRA 秩 |
| lora_alpha | 128 | = 2 × rank |
| min_group_std | 0.01 | 组内方差保护，std < 此值跳过更新 |

---

## 七、成本估算

| 项目 | 数量 | 费用 |
|------|------|------|
| 采样 (K=8 × 200 samples) | 1600 次 API | ~¥70 (qwen-max) / ~¥10 (自部署) |
| LLM Judge (30% 触发) | ~480 次 | ~¥20 |
| GPU 训练 (4×A100, 500 step) | ~25 小时 | ~$200-300 |
| **总计** | | **~$300-400** |

---

## 八、风险与应对

| 风险 | 概率 | 应对 |
|------|------|------|
| 基础模型排序能力太弱 (K=8 全是垃圾) | 20% | 快速验证（20 sample × 8 采样），不行就先做一轮 SFT warm start |
| Reward Hacking | 30% | 多维度交叉验证 + held-out 评测 + 人工抽检 |
| 训练 Collapse | 25% | min_std 保护 + 高 temperature + 频繁 checkpoint |
| Reward 权重不合理 | 30% | 验证集上 grid search 权重 |

---

## 九、实施步骤

| 步骤 | 任务 | 耗时 | 产出 |
|------|------|------|------|
| 1 | 数据清洗：对目标数据集跑 dataset_cleaner | 2-3 小时 | clean_pos/clean_neg 标签 |
| 2 | 快速验证：20 sample × 8 采样，检查 reward 分布 | 1 小时 | 确认模型 baseline 能力够用 |
| 3 | 全量采样：200 sample × 8 采样 | 2-4 小时 | data/grpo/grpo_train.jsonl |
| 4 | 环境搭建：GPU 集群 + OpenRLHF | 1-2 天 | Pipeline 跑通 |
| 5 | 小规模试训 (50 step) | 1 天 | 验证 reward 上升、无 collapse |
| 6 | 正式训练 (500 step) | 1-2 天 | 最佳 checkpoint |
| 7 | 离线评测 + 对比 | 0.5 天 | Pos Hit@1 ≥ 38% |
| **总计** | | **~1-2 周** | |
