"""
构建 mid 数据集：20,000 条，站点均衡，negative 优先

1. 从 v3 清洗好的数据中采样
2. 用完整 system prompt 构建正确的 prompt 格式
3. 6 个站点均衡分配
4. 优先保留有 negative_keys 的样本
"""

import pandas as pd
import numpy as np
import json
import oss2
from pathlib import Path
from collections import Counter

OUTPUT_DIR = Path("/tmp/rank_mid")
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET_TOTAL = 20000

# ─── 完整 System Prompt ─────────────────────────────────────────────
SYSTEM_PROMPT = """你是Lazada电商广告排序助手，名字叫Paimon。你的唯一任务是根据商家状态与广告卡片池，为池内每一张卡片进行推荐优先级的排序。

🔴【绝对输出铁律】（优先级最高，违反即判为错误）
1. 仅输出一个合法的 JSON 字符串数组，必须以 [ 开头，以 ] 结尾。
2. 严禁使用任何 Markdown 格式（绝对禁止输出 ```json 或 ```）。
3. 严禁输出任何开场白、结束语、推理过程、空格换行外的额外字符。
4. 数组必须严格按推荐优先级降序排列（优先级最高的卡片必须放在索引 0）。
5. 卡片池中每张卡片都必须输出，不遗漏、不新增、不修改原始 Key 名称。

---

# 一、基础知识

## 1.1 广告计划类型
| 类型 | 说明 |
|------|------|
| **SMAX AA** | 全站推广商品，全自动AI优化，高托管 |
| **SMAX AM** | 全站推广商品，手动出价，半托管 |
| **SMAX AM_X**（即 SMAX AM Campaign） | 全站推广店铺（多品模式），高托管；卡片中"多品SMAX AM"对应此类型 |
| **SD AA / SD AM** | 旧版推广，优先级低于SMAX类 |
| **MSA** | 大促爆单包，仅大促期有效 |

> ⚠️ **SMAX AM ≠ SMAX AM_X**，两者是不同计划类型，判断计划存在性时须精确匹配，严禁混用。

## 1.2 关键概念
| 概念 | 定义 |
|------|------|
| **OOBA** | 余额不足风险：`balance / l7d_avg_spend < 3` |
| **OOBU** | 预算撞线风险：当日预算使用率 ≥ 85% |
| **长期计划** | `end_date = 3020年`，无需推荐"延长投放时间" |
| **大促期** | 当前日期满足 `start_date ≤ 今日 ≤ end_date` 的MSA计划存在 |
| **rapid_boost（极速起量）** | 仅适用于 SMAX AA / SMAX AM_X，且该计划采用 troi 出价 |

---

# 二、核心决策规则

## 规则1：资金风险（最高优先级）
**OOBA风险**
- 有风险（`balance / l7d_avg_spend < 3`）→ 手动充值类卡片优先级极高（≥ 0.80）
- 无风险 → 充值类卡片优先级极低（≤ 0.10）

**OOBU风险**
- 有风险（预算使用率 ≥ 85%，你需要参考历史的预算使用率，以及当天是否预算花费很快，比如早上7点，预算已经使用了50%，当天很有可能预算有风险）→ **对应计划类型**的手动提升预算卡片高优先
- 无风险 → 对应计划类型的提预算卡片优先级极低（≤ 0.10）
- ⚠️ OOBU风险须**精确到计划类型**，不可跨类型套用

**自动化兜底建议**（自动充值、自动提升预算）优先级均极低，推荐须足够谨慎

## 规则2：计划存在性约束（硬约束，不可违背）
| 计划状态 | 对应卡片处理 |
|----------|-------------|
| **计划不存在** | 该类所有操作卡片（提预算/改出价/延长时间等）优先级极低 |
| **计划存在且活跃** | 创建类卡片优先级低；重激活类卡片优先级极低 |
| **计划存在但已关闭** | 重激活类卡片优先级提升 |
| **长期计划（end_date=3020年）** | 延长投放时间类卡片优先级极低 |

> 示例：商家仅有 SD AM 计划 → 所有 `smax_aa_`、`sd_aa_`、`msa_*` 卡片优先级接近0

## 规则3：上下文硬约束
| 条件 | 约束效果 |
|------|---------|
| `is_mega_period = false` | 所有 MSA 相关卡片优先级极低 |
| `problem_type = "Low ROI"` | rapid_boost 类卡片优先级极低 |
| 某卡片历史曝光 ≥ 3次且均未被采纳 | 优先级显著降低 |

## 规则4：商家近期行为意图
从 `trigger_action` 中提取商家近期操作，识别主动意图：
| 近期行为 | 判断意图 | 对应降权卡片 |
|----------|---------|-------------|
| 提高 troi / iopRoas | 控成本 | rapid_boost类、扩量类、降低troi类 |
| 降低 troi / iopRoas | 扩量 | 提升 rapid_boost、扩量类优先级 |
| 提高预算 | 主动扩量 | 降低"降低预算"类卡片 |
| 降低预算 | 控制花费 | 降低"提高预算"类卡片优先级 |

> 行为与卡片建议方向相悖 → 该卡片优先级显著降低

## 规则5：场景加权
**大促期**
- MSA操作类 > SMAX同类操作

**广告表现不佳**（竞价胜率低 / 点击率低 / 转化差）
- 出价优化类卡片优先级提升

---

# 三、推荐优先级参考顺序
1. 资金/预算风险修复
   └─ OOBA手动充值 > OOBU手动提预算（精确到计划类型）
2. 大促期特殊操作（is_mega_period = true）
   └─ MSA优化 > MSA创建 > 其他
3. 广告计划优化
   └─ 预算闲置时：优先调整出价（max_bid / troi）
   └─ 预算为主要限制时：优先提升Budget
   └─ 出价/预算调整 > 调整为BCB > 增加投放品
   └─ 预算设置为unlimited时，不应该有提升Budget建议
4. 新计划创建
   └─ SMAX AA / SMAX AM_X / SMAX AM > SD类
5. 自动化辅助（需用户高度信任，优先级低）
   └─ 自动充值 / 自动提升预算
6. 不适用操作（优先级极低）
   └─ 计划不存在 / 硬约束命中 / 重复未采纳

> ⚠️ 以上为方向参考，需结合商家实际状态综合判断，不可机械套用。

---

# 四、评分前强制检查清单
在生成排序前，请在内部逐项确认（无需输出）：
□ 1. 商家实际拥有哪些计划类型？
□ 2. 是否存在 OOBA 风险？（balance / l7d_avg_spend < 3）
□ 3. 各计划类型是否存在 OOBU 风险？（预算使用率 ≥ 85%）
□ 4. is_mega_period 是否为 true？
□ 5. has_rapid_boost_enabled 是否为 true？
□ 6. 各计划的 end_date 是否为 3020年？
□ 7. 商家近期 trigger_action 反映哪些行为意图？
□ 8. 是否有卡片历史曝光 ≥ 3次且均未被采纳？

---

# 五、最终输出格式
严格按以下标准 JSON 数组格式输出，无需任何额外字符、注释或换行：
["卡片1_Key", "卡片2_Key", "卡片3_Key", ...]"""


def build_prompt(gt):
    """构建完整 prompt"""
    ctx = gt['context']

    try:
        campaign_data = json.loads(ctx.get('campaign_data', '{}')) if isinstance(ctx.get('campaign_data'), str) else ctx.get('campaign_data', {})
    except:
        campaign_data = {}

    is_mega = "今日有大促" if campaign_data.get('summary', {}).get('msa_count', 0) > 0 else "今日无大促"

    user_part = f"""请帮我根据商家特征给出推荐的概率(0-1之间的数值)对以下广告卡片池进行排序，按推荐概率从高到低排列。

## 当前日期:
{ctx.get('rank_timestamp', '')}

## 商家近期行为记录:
{ctx.get('seller_behavior', '无')}

## 广告卡片池:{json.dumps(gt['card_pool'], ensure_ascii=False)}

## 商家相关信息:
### 今日大促信息:
{is_mega}

### 商家当前钱包数据:
{ctx.get('wallet_data', '{}')}

### 商家广告投放效果：
{ctx.get('campaign_data', '{}')}"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_part},
    ]


# ─── 1. 加载 v3 数据 ──────────────────────────────────────────────────
print("=== 1. 加载 v3 数据 ===")
train_v3 = pd.read_parquet("/tmp/rank_cleaned_v3/train.parquet")
test_v3 = pd.read_parquet("/tmp/rank_cleaned_v3/test.parquet")
all_data = pd.concat([train_v3, test_v3], ignore_index=True)
print(f"总样本: {len(all_data)}")

all_data['venture'] = all_data['ground_truth'].apply(lambda x: json.loads(x)['context']['venture'])
all_data['has_neg'] = all_data['ground_truth'].apply(lambda x: len(json.loads(x).get('negative_keys', [])) > 0)

# ─── 2. 站点均衡采样 ─────────────────────────────────────────────────
print(f"\n=== 2. 站点均衡采样 (目标: {TARGET_TOTAL}) ===")

ventures = all_data['venture'].value_counts()
print(f"原始分布:\n{ventures}\n")

n_ventures = len(ventures)
per_venture = TARGET_TOTAL // n_ventures
remainder = TARGET_TOTAL - per_venture * n_ventures
print(f"每站点目标: {per_venture} (+余数 {remainder})\n")

sampled = []
for i, (venture, _) in enumerate(ventures.items()):
    vdf = all_data[all_data['venture'] == venture].copy()
    target = per_venture + (1 if i < remainder else 0)

    has_neg = vdf[vdf['has_neg']]
    no_neg = vdf[~vdf['has_neg']]

    if len(has_neg) >= target:
        # negative 样本够多，全要 negative + 随机补
        neg_s = has_neg.sample(target, random_state=42)
        s = neg_s
    else:
        # 全要 negative，用 no_neg 补满
        bc_needed = target - len(has_neg)
        bc_s = no_neg.sample(min(bc_needed, len(no_neg)), random_state=42)
        s = pd.concat([has_neg, bc_s])

    actual = len(s)
    neg_count = s['has_neg'].sum()
    print(f"  {venture}: {len(vdf)} → {actual} (negative: {neg_count}, {neg_count/actual*100:.0f}%)")
    sampled.append(s)

result = pd.concat(sampled)
print(f"\n采样后总数: {len(result)}")

# ─── 3. 构建 prompt ──────────────────────────────────────────────────
print(f"\n=== 3. 构建 prompt ===")

rows = []
for _, row in result.iterrows():
    gt = json.loads(row['ground_truth'])
    prompt = build_prompt(gt)
    rows.append({
        'prompt': prompt,
        'reward_model': {'style': 'rule', 'ground_truth': row['ground_truth']},
        'data_source': 'ranking',
    })

final_df = pd.DataFrame(rows)

# ─── 4. 划分 8:2 ─────────────────────────────────────────────────────
print(f"\n=== 4. 划分 train/test ===")
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
split = int(len(final_df) * 0.8)
train_df = final_df.iloc[:split]
test_df = final_df.iloc[split:]
print(f"训练集: {len(train_df)} ({len(train_df)/len(final_df)*100:.1f}%)")
print(f"验证集: {len(test_df)} ({len(test_df)/len(final_df)*100:.1f}%)")

# ─── 5. 保存 ─────────────────────────────────────────────────────────
train_path = OUTPUT_DIR / "train.parquet"
test_path = OUTPUT_DIR / "test.parquet"
train_df.to_parquet(train_path, index=False)
test_df.to_parquet(test_path, index=False)
print(f"\n训练集: {train_path} ({train_path.stat().st_size / 1024 / 1024:.1f} MB)")
print(f"验证集: {test_path} ({test_path.stat().st_size / 1024 / 1024:.1f} MB)")

# ─── 6. 质量验证 ─────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"=== 6. 质量验证 ===")
print(f"{'='*60}")

for name, path in [("训练集", train_path), ("验证集", test_path)]:
    df_check = pd.read_parquet(path)
    gts = [json.loads(x['ground_truth']) if isinstance(x, dict) else json.loads(x) for x in df_check['reward_model']]

    pos_lens = [len(g['positive_actions']) for g in gts]
    neg_lens = [len(g['negative_keys']) for g in gts]
    pool_lens = [len(g['card_pool']) for g in gts]
    ventures = [g['context']['venture'] for g in gts]

    print(f"\n{name} ({len(df_check)} 样本):")
    print(f"  positive_actions: 非空={sum(1 for x in pos_lens if x>0)} ({sum(1 for x in pos_lens if x>0)/len(pos_lens)*100:.1f}%), avg={np.mean(pos_lens):.2f}")
    print(f"    分布: {dict(sorted(Counter(pos_lens).items()))}")
    print(f"  negative_keys:    非空={sum(1 for x in neg_lens if x>0)} ({sum(1 for x in neg_lens if x>0)/len(neg_lens)*100:.1f}%), avg={np.mean(neg_lens):.2f}")
    print(f"    分布: {dict(sorted(Counter(neg_lens).items()))}")
    print(f"  card_pool:        avg={np.mean(pool_lens):.1f}, min={min(pool_lens)}, max={max(pool_lens)}")
    print(f"  venture:          {dict(sorted(Counter(ventures).items()))}")

    invalid = sum(1 for g in gts if any(a not in g['card_pool'] for a in g['positive_actions']))
    print(f"  positive 不在 pool 中: {invalid}")

    pools = [tuple(sorted(g['card_pool'])) for g in gts]
    print(f"  唯一 card_pool 组合: {len(set(pools))}")

    prompts = [list(x) if hasattr(x, '__iter__') and not isinstance(x, str) else x for x in df_check['prompt']]
    avg_chars = np.mean([len(json.dumps(list(p) if hasattr(p, 'tolist') else p, ensure_ascii=False)) for p in prompts])
    print(f"  prompt 平均长度: {avg_chars:.0f} chars")

# ─── 7. GRPO 覆盖分析 ───────────────────────────────────────────────
train_len = len(train_df)
consumed = 32 * 500  # batch_size * total_steps
print(f"\n{'='*60}")
print(f"=== 7. GRPO 覆盖分析 ===")
print(f"{'='*60}")
print(f"  训练集: {train_len} 样本")
print(f"  500 步消耗: {consumed} prompt")
print(f"  每样本平均被采样: {consumed/train_len:.1f} 次")
if train_len >= consumed:
    print(f"  ✅ 数据充足，500 步不需要重复")
else:
    print(f"  ⚠️  每样本被重复 {consumed/train_len:.1f} 次")

# ─── 8. 上传到 OSS ──────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"=== 8. 上传到 OSS ===")
print(f"{'='*60}")

ACCESS_KEY_ID = os.environ.get("OSS_ACCESS_ID", "")
ACCESS_KEY_SECRET = os.environ.get("OSS_ACCESS_KEY", "")
ENDPOINT = "https://oss-cn-hangzhou-zmf.aliyuncs.com"
BUCKET_NAME = "lazada-ai-model"

auth = oss2.Auth(ACCESS_KEY_ID, ACCESS_KEY_SECRET)
bucket = oss2.Bucket(auth, ENDPOINT, BUCKET_NAME)

files = [
    (str(train_path), "ad/guoshauile.gsl/data/midrank/train.parquet"),
    (str(test_path), "ad/guoshauile.gsl/data/midrank/test.parquet"),
]

for local_path, oss_path in files:
    size_mb = Path(local_path).stat().st_size / 1024 / 1024
    print(f"  {local_path} ({size_mb:.1f} MB) -> oss://{BUCKET_NAME}/{oss_path}")
    r = bucket.put_object_from_file(oss_path, local_path)
    print(f"  Status: {r.status} {'✅' if r.status == 200 else '❌'}")

print(f"\n{'='*60}")
print(f"✅ 完成!")
print(f"{'='*60}")
print(f"\n数据集对比:")
print(f"  minirank:  9,909 样本 (站点均衡, 快速验证)")
print(f"  midrank:   {len(final_df)} 样本 (站点均衡, 正式训练)")
print(f"  rank:     50,000 样本 (全量, 大规模训练)")
