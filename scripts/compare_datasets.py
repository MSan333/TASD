"""
对比 ranking 和 minirank 数据集

从 OSS 下载两个数据集并对比:
  - 样本数量
  - prompt 格式
  - positive/negative 分布
  - card_pool 分布
  - venture 分布
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from collections import Counter

OUTPUT_DIR = Path("/tmp/dataset_compare")
OUTPUT_DIR.mkdir(exist_ok=True)

# OSS 路径
OSS_ROOT = "/data/oss_bucket_0/ad/guoshauile.gsl"
DATASETS = {
    "ranking": {
        "train": f"{OSS_ROOT}/data/ranking/train.parquet",
        "test": f"{OSS_ROOT}/data/ranking/test.parquet",
    },
    "minirank": {
        "train": f"{OSS_ROOT}/data/minirank/train.parquet",
        "test": f"{OSS_ROOT}/data/minirank/test.parquet",
    },
}


def analyze_dataset(name, train_path, test_path):
    """分析单个数据集"""
    print(f"\n{'='*70}")
    print(f"数据集: {name}")
    print(f"{'='*70}")

    stats = {}

    for split, path in [("train", train_path), ("test", test_path)]:
        try:
            df = pd.read_parquet(path)
        except Exception as e:
            print(f"  [{split}] 读取失败: {e}")
            continue

        print(f"\n[{split}] {len(df)} 样本")

        # 1. prompt 格式分析
        print(f"\n  Prompt 格式:")
        prompt_types = []
        prompt_lens = []
        has_system = 0
        has_detailed_rules = 0

        for prompt in df['prompt']:
            if isinstance(prompt, (list, np.ndarray)):
                # messages 格式
                prompt_types.append("messages")
                if len(prompt) >= 2 and prompt[0].get('role') == 'system':
                    has_system += 1
                    system_content = prompt[0].get('content', '')
                    if '绝对输出铁律' in system_content or '规则1' in system_content:
                        has_detailed_rules += 1
                total_len = sum(len(json.dumps(m, ensure_ascii=False)) for m in prompt)
                prompt_lens.append(total_len)
            else:
                # string 格式
                prompt_types.append("string")
                prompt_lens.append(len(prompt))

        print(f"    类型分布: {dict(Counter(prompt_types))}")
        print(f"    有 system role: {has_system} ({has_system/len(df)*100:.1f}%)")
        print(f"    有详细规则: {has_detailed_rules} ({has_detailed_rules/len(df)*100:.1f}%)")
        print(f"    平均长度: {np.mean(prompt_lens):.0f} chars")
        print(f"    长度范围: [{min(prompt_lens)}, {max(prompt_lens)}]")

        # 2. ground_truth 分析
        print(f"\n  Ground Truth:")
        if 'ground_truth' in df.columns:
            gt_col = 'ground_truth'
        elif 'reward_model' in df.columns:
            gt_col = 'reward_model'
        else:
            print(f"    未找到 ground_truth 列")
            continue

        gts = []
        for x in df[gt_col]:
            if isinstance(x, dict):
                gts.append(json.loads(x['ground_truth']))
            else:
                gts.append(json.loads(x))

        pos_lens = [len(g['positive_actions']) for g in gts]
        neg_lens = [len(g['negative_keys']) for g in gts]
        pool_lens = [len(g['card_pool']) for g in gts]
        ventures = [g['context']['venture'] for g in gts]

        print(f"    positive_actions:")
        print(f"      非空: {sum(1 for x in pos_lens if x>0)} ({sum(1 for x in pos_lens if x>0)/len(pos_lens)*100:.1f}%)")
        print(f"      分布: {dict(sorted(Counter(pos_lens).items()))}")
        print(f"      avg: {np.mean(pos_lens):.2f}")

        print(f"    negative_keys:")
        print(f"      非空: {sum(1 for x in neg_lens if x>0)} ({sum(1 for x in neg_lens if x>0)/len(neg_lens)*100:.1f}%)")
        print(f"      分布: {dict(sorted(Counter(neg_lens).items()))}")
        print(f"      avg: {np.mean(neg_lens):.2f}")

        print(f"    card_pool:")
        print(f"      avg: {np.mean(pool_lens):.1f}, min: {min(pool_lens)}, max: {max(pool_lens)}")
        print(f"      分布: {dict(sorted(Counter(pool_lens).items()))}")

        print(f"    venture:")
        print(f"      分布: {dict(sorted(Counter(ventures).items()))}")

        # 3. 质量检查
        invalid = sum(1 for g in gts if any(a not in g['card_pool'] for a in g['positive_actions']))
        print(f"\n  质量:")
        print(f"    positive 不在 pool 中: {invalid}")

        pools = [tuple(sorted(g['card_pool'])) for g in gts]
        print(f"    唯一 card_pool 组合: {len(set(pools))}")

        stats[split] = {
            'n_samples': len(df),
            'prompt_type': Counter(prompt_types).most_common(1)[0][0],
            'has_system': has_system,
            'has_detailed_rules': has_detailed_rules,
            'prompt_len_mean': np.mean(prompt_lens),
            'positive_non_empty': sum(1 for x in pos_lens if x>0) / len(pos_lens),
            'negative_non_empty': sum(1 for x in neg_lens if x>0) / len(neg_lens),
            'pool_len_mean': np.mean(pool_lens),
            'n_unique_pools': len(set(pools)),
            'ventures': dict(Counter(ventures)),
        }

    return stats


# ─── 分析两个数据集 ──────────────────────────────────────────────────
print("开始分析数据集...")

all_stats = {}
for name, paths in DATASETS.items():
    stats = analyze_dataset(name, paths['train'], paths['test'])
    all_stats[name] = stats

# ─── 对比总结 ──────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("对比总结")
print(f"{'='*70}")

for split in ['train', 'test']:
    print(f"\n[{split}]")
    for name in DATASETS.keys():
        if split in all_stats.get(name, {}):
            s = all_stats[name][split]
            print(f"\n  {name}:")
            print(f"    样本数: {s['n_samples']}")
            print(f"    Prompt 类型: {s['prompt_type']}")
            print(f"    有 system role: {s['has_system']} ({s['has_system']/s['n_samples']*100:.1f}%)")
            print(f"    有详细规则: {s['has_detailed_rules']} ({s['has_detailed_rules']/s['n_samples']*100:.1f}%)")
            print(f"    Prompt 平均长度: {s['prompt_len_mean']:.0f}")
            print(f"    Positive 非空率: {s['positive_non_empty']*100:.1f}%")
            print(f"    Negative 非空率: {s['negative_non_empty']*100:.1f}%")
            print(f"    Pool 平均大小: {s['pool_len_mean']:.1f}")
            print(f"    唯一 pool 数: {s['n_unique_pools']}")

print(f"\n{'='*70}")
print("✅ 分析完成")
print(f"{'='*70}")
