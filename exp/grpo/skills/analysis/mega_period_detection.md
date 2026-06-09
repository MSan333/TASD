---
name: mega_period_detection
description: 从campaign_data中判断当前是否处于大促期，决定MSA卡片优先级
layer: analysis
requires_input:
  - campaign_data
enabled: true
version: v1
---

# 大促期判断

## 任务
从 campaign_data 中检查 MSA（大促爆单包）计划的存在性和时间有效性，判断当前是否处于大促期。

## 分析步骤

### 1. 查找 MSA 计划
在 campaign_data 中搜索类型为 MSA 的计划。

### 2. 判断时间有效性
如果存在 MSA 计划，检查其 start_date 和 end_date：
- 当前日期在 [start_date, end_date] 区间内 → 大促期
- 当前日期不在区间内 → 非大促期
- 无 MSA 计划 → 非大促期

### 3. 输出判定
| 条件 | 判定 | 对排序的影响 |
|------|------|------------|
| MSA 活跃且在有效期内 | 大促期 | MSA 操作类卡片优先级大幅提升 |
| MSA 存在但已过期/未开始 | 非大促期 | MSA 卡片优先级极低 |
| 无 MSA 计划 | 非大促期 | 所有 MSA 相关卡片优先级极低 |

## 输出格式
一句话：`大促：{是/否}，MSA状态：{具体状态}`
例：`大促：否，MSA状态：无MSA计划`
例：`大促：是，MSA状态：MSA×2活跃(06.01-06.15)`
