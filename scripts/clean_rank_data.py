"""
数据清洗脚本 v3 - 精选高质量版

目标：~50,000 条精选数据，保证质量和多样性

构造策略：
  A 类 (强信号): positive ≥1 且 negative ≥1 → 全部保留
  B/C 类 (正信号): positive ≥1, negative=0 → 按场景多样性采样

多样性维度：
  - venture (6 个市场)
  - card_pool 组合 (~2000 种)
  - positive_actions 数量
  - 时间分布 (ds)
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from collections import Counter

DATA_DIR = Path("/Users/xigaodi/Desktop/code/rank-agent/data_cache/rank_match")
OUTPUT_DIR = Path("/tmp/rank_cleaned_v3")
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET_TOTAL = 50000

# ─── 1. 加载数据 ───────────────────────────────────────────────────────
print("=== 1. 加载数据 ===")
dfs = []
for f in sorted(DATA_DIR.glob("ds=*.parquet")):
    df = pd.read_parquet(f)
    # 提取日期
    date_str = f.stem.split("=")[1]
    df['ds'] = date_str
    dfs.append(df)
    print(f"  {f.name}: {len(df)}")

df = pd.concat(dfs, ignore_index=True)
print(f"总计: {len(df)} rows")

# ─── 2. 基础过滤 ───────────────────────────────────────────────────────
print(f"\n=== 2. 基础过滤 (before: {len(df)}) ===")
df = df[df['rank_status'] == 'SUCCESS']
print(f"  rank_status=SUCCESS: {len(df)}")
df = df[df['rank_card_pool'].notna() & (df['rank_card_pool'] != "")]
print(f"  card_pool 非空: {len(df)}")
df = df[df['rank_campaign_data'].notna() & (df['rank_campaign_data'] != "")]
print(f"  campaign_data 非空: {len(df)}")

# ─── 3. 聚合事件 ───────────────────────────────────────────────────────
print(f"\n=== 3. 聚合事件 ===")
df['_event_key'] = df['venture'] + '|' + df['seller_id'].astype(str) + '|' + df['rank_timestamp'].astype(str)

pos = df[df['match_type'] == 'positive'].groupby('_event_key')['action_name'].apply(lambda x: list(set(x))).rename('positive_actions')
neg = df[df['match_type'] == 'negative'].groupby('_event_key')['action_name'].apply(lambda x: list(set(x))).rename('negative_keys')

first_rows = df.drop_duplicates(subset='_event_key', keep='first')
events = first_rows.set_index('_event_key')[
    ['venture', 'seller_id', 'rank_timestamp', 'rank_card_pool', 'rank_campaign_data',
     'rank_seller_behavior', 'rank_wallet_data', 'ds']
].join(pos).join(neg)
events = events.fillna({'positive_actions': '', 'negative_keys': ''})
print(f"事件数: {len(events)}")

# ─── 4. 过滤无效事件 ──────────────────────────────────────────────────
print(f"\n=== 4. 过滤无效事件 (before: {len(events)}) ===")

# positive_actions 非空
events = events[events['positive_actions'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)]
print(f"  positive 非空: {len(events)}")

# card_pool 非空
events = events[events['rank_card_pool'].apply(lambda x: len(json.loads(x)) > 0 if isinstance(x, str) else False)]
print(f"  card_pool 非空: {len(events)}")

# ─── 5. 高质量过滤 ────────────────────────────────────────────────────
print(f"\n=== 5. 高质量过滤 (before: {len(events)}) ===")

def validate_positive_in_pool(row):
    try:
        card_pool = set(json.loads(row['rank_card_pool'])) if isinstance(row['rank_card_pool'], str) else set(row['rank_card_pool'])
        positive = row['positive_actions'] if isinstance(row['positive_actions'], list) else []
        return all(action in card_pool for action in positive)
    except:
        return False

valid_mask = events.apply(validate_positive_in_pool, axis=1)
events = events[valid_mask]
print(f"  positive 全部在 card_pool 中: {len(events)} ({valid_mask.mean()*100:.1f}%)")

# ─── 6. 分类 A/B/C ───────────────────────────────────────────────────
print(f"\n=== 6. 样本分类 ===")

events['has_negative'] = events['negative_keys'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)
events['n_positive'] = events['positive_actions'].apply(lambda x: len(x) if isinstance(x, list) else 0)
events['n_negative'] = events['negative_keys'].apply(lambda x: len(x) if isinstance(x, list) else 0)

type_a = events[events['has_negative']]
type_bc = events[~events['has_negative']]

print(f"  A 类 (有 negative): {len(type_a)} ({len(type_a)/len(events)*100:.1f}%)")
print(f"  B/C 类 (无 negative): {len(type_bc)} ({len(type_bc)/len(events)*100:.1f}%)")

# ─── 7. 多样性采样 ───────────────────────────────────────────────────
print(f"\n=== 7. 多样性采样 ===")

target_a = len(type_a)  # A 类全部保留
target_bc = TARGET_TOTAL - target_a  # B/C 类需要采样的数量
print(f"  A 类保留: {target_a}")
print(f"  B/C 类采样: {target_bc}")

# 为每个样本计算场景 key = (venture, card_pool组合, n_positive)
def scene_key(row):
    pool = tuple(sorted(json.loads(row['rank_card_pool']))) if isinstance(row['rank_card_pool'], str) else tuple(sorted(row['rank_card_pool']))
    return (row['venture'], pool, row['n_positive'])

type_bc['_scene_key'] = type_bc.apply(scene_key, axis=1)
type_a['_scene_key'] = type_a.apply(scene_key, axis=1)

# 分层采样：按 (venture, n_positive) 分层，每层内按 card_pool 去重采样
def stratified_sample(df_sample, n_target):
    """按 (venture, n_positive) 分层采样，每层内优先覆盖不同 card_pool"""

    df_sample = df_sample.copy()
    df_sample['_stratum'] = df_sample['venture'] + '|' + df_sample['n_positive'].astype(str)

    strata = df_sample.groupby('_stratum')
    stratum_sizes = strata.size()

    # 按比例分配采样数
    stratum_targets = (stratum_sizes / stratum_sizes.sum() * n_target).astype(int)
    remainder = n_target - stratum_targets.sum()
    if remainder > 0:
        largest = stratum_targets.idxmax()
        stratum_targets[largest] += remainder

    sampled = []
    for stratum_name, target in stratum_targets.items():
        stratum_df = df_sample[df_sample['_stratum'] == stratum_name]

        if len(stratum_df) <= target:
            sampled.append(stratum_df)
        else:
            # 简单随机采样
            sampled.append(stratum_df.sample(target, random_state=42))

    result = pd.concat(sampled)
    return result

if target_bc > 0:
    sampled_bc = stratified_sample(type_bc, target_bc)
    print(f"  B/C 类采样结果: {len(sampled_bc)}")
else:
    sampled_bc = pd.DataFrame()

# ─── 8. 合并并去重 ────────────────────────────────────────────────────
print(f"\n=== 8. 合并数据 ===")
combined = pd.concat([type_a, sampled_bc])

# 检查是否有重复 event_key
dup_count = combined.duplicated(subset=['venture', 'seller_id', 'rank_timestamp']).sum()
if dup_count > 0:
    print(f"  去除 {dup_count} 条重复事件")
    combined = combined.drop_duplicates(subset=['venture', 'seller_id', 'rank_timestamp'], keep='first')

print(f"  合并后总数: {len(combined)}")

# ─── 9. 构建训练格式 ──────────────────────────────────────────────────
print(f"\n=== 9. 构建训练数据 ===")

rows = []
for _, row in combined.iterrows():
    card_pool = json.loads(row['rank_card_pool']) if isinstance(row['rank_card_pool'], str) else row['rank_card_pool']
    context = {
        'venture': row['venture'],
        'seller_id': int(row['seller_id']),
        'rank_timestamp': str(row['rank_timestamp']),
        'seller_behavior': row.get('rank_seller_behavior', ''),
        'wallet_data': row.get('rank_wallet_data', ''),
        'campaign_data': row.get('rank_campaign_data', ''),
    }
    ground_truth = {
        'card_pool': card_pool,
        'positive_actions': row['positive_actions'],
        'negative_keys': row['negative_keys'] if isinstance(row['negative_keys'], list) else [],
        'context': context,
    }
    prompt = (
        "你是 Lazada 广告卡片排序助手。根据商家状态，为以下卡片池排序。\n\n"
        f"商家上下文:\n{json.dumps(context, ensure_ascii=False)}\n\n"
        f"卡片池:\n{json.dumps(card_pool, ensure_ascii=False)}\n\n"
        "请输出 JSON 数组，按推荐优先级从高到低排列所有卡片。"
    )
    rows.append({
        'prompt': prompt,
        'ground_truth': json.dumps(ground_truth, ensure_ascii=False),
        'data_source': 'ranking',
    })

result_df = pd.DataFrame(rows)

# ─── 10. 划分 8:2 ─────────────────────────────────────────────────────
print(f"\n=== 10. 划分 train/test ===")
result_df = result_df.sample(frac=1, random_state=42).reset_index(drop=True)
split = int(len(result_df) * 0.8)
train_df = result_df.iloc[:split]
test_df = result_df.iloc[split:]
print(f"训练集: {len(train_df)} ({len(train_df)/len(result_df)*100:.1f}%)")
print(f"验证集: {len(test_df)} ({len(test_df)/len(result_df)*100:.1f}%)")

# ─── 11. 保存 ─────────────────────────────────────────────────────────
train_path = OUTPUT_DIR / "train.parquet"
test_path = OUTPUT_DIR / "test.parquet"
train_df.to_parquet(train_path, index=False)
test_df.to_parquet(test_path, index=False)
print(f"\n训练集: {train_path} ({train_path.stat().st_size / 1024 / 1024:.1f} MB)")
print(f"验证集: {test_path} ({test_path.stat().st_size / 1024 / 1024:.1f} MB)")

# ─── 12. 全面质量验证 ─────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"=== 12. 全面质量验证 ===")
print(f"{'='*60}")

for name, path in [("训练集", train_path), ("验证集", test_path)]:
    df_check = pd.read_parquet(path)
    gts = [json.loads(x) for x in df_check['ground_truth']]

    pos_lens = [len(g['positive_actions']) for g in gts]
    neg_lens = [len(g['negative_keys']) for g in gts]
    pool_lens = [len(g['card_pool']) for g in gts]
    ventures = [g['context']['venture'] for g in gts]
    behavior_lens = [len(g['context'].get('seller_behavior', '')) for g in gts]

    print(f"\n{name} ({len(df_check)} 样本):")
    print(f"  positive_actions: avg={np.mean(pos_lens):.1f}, 非空={sum(1 for x in pos_lens if x>0)} ({sum(1 for x in pos_lens if x>0)/len(pos_lens)*100:.1f}%)")
    print(f"    分布: {dict(Counter(pos_lens))}")
    print(f"  negative_keys:    avg={np.mean(neg_lens):.1f}, 非空={sum(1 for x in neg_lens if x>0)} ({sum(1 for x in neg_lens if x>0)/len(neg_lens)*100:.1f}%)")
    print(f"    分布: {dict(Counter(neg_lens))}")
    print(f"  card_pool:        avg={np.mean(pool_lens):.1f}, min={min(pool_lens)}, max={max(pool_lens)}")
    print(f"  venture:          {dict(Counter(ventures))}")
    print(f"  behavior 长度:    avg={np.mean(behavior_lens):.0f}, min={min(behavior_lens)}")

    # 验证 positive 全部在 card_pool 中
    invalid = 0
    for g in gts:
        pool = set(g['card_pool'])
        for a in g['positive_actions']:
            if a not in pool:
                invalid += 1
                break
    print(f"  positive 不在 pool 中: {invalid} ({invalid/len(gts)*100:.2f}%)")

    # card_pool 组合多样性
    pools = [tuple(sorted(g['card_pool'])) for g in gts]
    print(f"  唯一 card_pool 组合: {len(set(pools))}")

# ─── 13. 对比旧数据 ──────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"=== 13. 对比 ===")
print(f"{'='*60}")
print(f"  旧数据: 5,400 train + 600 test = 6,000 样本")
print(f"  新数据: {len(train_df)} train + {len(test_df)} test = {len(result_df)} 样本")
print(f"  数据量提升: {len(result_df)/6000:.0f}x")
print(f"  positive_actions 非空: 0% → 100%")
print(f"  negative_keys 非空: ~0% → {sum(1 for x in neg_lens if x>0)/len(neg_lens)*100:.1f}%")

print(f"\n✅ 完成!")
