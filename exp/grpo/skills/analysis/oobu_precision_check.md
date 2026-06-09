---
name: oobu_precision_check
description: 逐计划类型检查OOBU预算撞线风险，输出精确到类型的风险清单
layer: analysis
requires_input:
  - campaign_data
enabled: true
version: v1
---

# OOBU 精确风险检查

## 任务
逐计划类型检查预算使用率，输出每个计划类型是否存在 OOBU（预算撞线）风险。排序时 OOBU 风险卡片必须精确到对应计划类型，不可跨类型套用。

## 分析步骤

### 1. 按类型聚合预算使用率
从 campaign_data 中按计划类型（SMAX AA / SMAX AM / SMAX AM_X / SD AA / SD AM / MSA）分别计算：
- 当日预算使用率 = 已消耗 / 日预算
- 如果有历史预算使用率，结合趋势判断

### 2. 风险判定
| 使用率 | 风险等级 | 说明 |
|--------|---------|------|
| ≥ 85% | 高风险 | OOBU 确认，对应类型提预算卡片高优 |
| 60%-85% + 上午时段消耗快 | 中风险 | 可能撞线，需关注 |
| < 60% | 无风险 | 对应类型提预算卡片优先级极低 |
| 预算=unlimited | 无风险 | 不存在 OOBU，不应推提预算 |

### 3. 关键约束
- OOBU 风险和对应的"提预算"卡片必须**类型匹配**：SMAX AM 有 OOBU → 只有 smax_am_budget_increase 高优，smax_aa_budget_increase 不受影响
- 预算设为 unlimited 的计划不存在 OOBU 风险

## 输出格式
一句话：`OOBU：{有风险的类型列表/无风险}`
例：`OOBU：SMAX AM(92%)高风险，其余无风险`
例：`OOBU：无风险（SMAX AA unlimited，SMAX AM 45%）`
