---
name: intent_recognition
description: 从商家近期行为日志中识别经营意图方向
layer: analysis
requires_input:
  - seller_behavior
enabled: true
version: v1
---

# 商家经营意图识别

## 任务
从 seller_behavior 行为日志中识别商家的经营意图。

## 分析步骤

### 1. 提取参数调整方向
找出 "from X to Y" 模式，判断各参数的升降：
- iopRoas/troi 下调 → 扩量信号（降低门槛拿更多流量）
- iopRoas/troi 上调 → 控本信号（提高门槛限制花费）
- budget 上调 → 扩量信号
- budget 下调 → 收缩信号
- maxBid/max_bid 上调 → 积极竞价
- maxBid/max_bid 下调 → 保守竞价

**注意交叉判断**：
- 降ROAS + 提预算 = 强扩量
- 降ROAS + 降预算 = 控本（不是扩量！降ROAS是为了降成本）
- 升ROAS + 降预算 = 强收缩
- 升ROAS + 提预算 = 扩量提效

### 2. 统计关键动作
- Create New Campaign → 扩量/冷启动
- Remove / OFF → 收缩/关停
- Manual TopUp → 资金补充，准备投放
- troiGuarantee Inactive → 关闭保障，可能在收缩
- 连续同类操作 ≥ 3 次 → 意图强度高

### 3. 综合判断
| 信号组合 | 判定意图 | 置信度 |
|---------|---------|--------|
| Create + 降ROAS + 提预算 | 强扩量 | 高 |
| 降ROAS + 降预算 | 控本 | 高 |
| 升ROAS + 降预算 + OFF | 强收缩 | 高 |
| 频繁调参但无方向性 | 调优/探索 | 中 |
| 无行为或仅TopUp | 观望/冷启动 | 低 |
| TopUp + Create | 冷启动扩量 | 高 |

## 输出格式
一句话：`意图：{方向}（{置信度}），关键信号：{信号列表}`
例：`意图：扩量（高置信），关键信号：Create×3, 降ROAS×2, 预算上调`
