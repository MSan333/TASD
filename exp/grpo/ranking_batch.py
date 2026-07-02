"""Batch reward function for GRPO ranking task (v11).

Called by: verl/utils/reward_score/feedback/__init__.py (compute_score_batch)
Interface: compute_score(data_sources, solution_strs, ground_truths, extra_infos) -> List[dict]

Returns List[dict] with "score" + sub-metrics
so BatchRewardManager populates reward_extra_info → SwanLab charts + console logs.

v11: Pairwise Preference Reward
  - 确定性 base_score (rule_score) 作为主信号
  - Pairwise Judge 比较组内 rollout, 输出离散 preference {-1, 0, +1}
  - 解决 v10 Judge 无区分度问题: 相对比较强制离散输出, 天然有方差
  - 保留 progressive invalid + clean bonus (v10 有效做法)

公式: score = base_score + α × judge_preference + format_penalty + clean_bonus
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from .reward_fn import (
    _parse_ranking,
    _detect_format_issues,
    _compute_format_penalty,
    _rule_based_score,
    format_gate,
)

logger = logging.getLogger(__name__)


def compute_score(
    data_sources: List[str],
    solution_strs: List[str],
    ground_truths: List[Any],
    extra_infos: Optional[List[Dict]] = None,
    **kwargs,
) -> List:
    """Batch compute_score compatible with veRL BatchRewardManager.verify().

    Returns List[dict] with "score" + sub-metrics (result_prediction, etc.)
    so BatchRewardManager populates reward_extra_info for SwanLab logging.
    """
    batch_size = len(solution_strs)
    if extra_infos is None:
        extra_infos = [{}] * batch_size

    use_batch_llm = bool(os.environ.get("OPENAI_API_KEY"))

    if use_batch_llm:
        results = _batch_with_llm(solution_strs, ground_truths, extra_infos)
    else:
        results = _batch_no_llm(solution_strs, ground_truths, extra_infos)

    # Logging summary
    scores_only = [r["score"] if isinstance(r, dict) else r for r in results]
    avg_score = sum(scores_only) / max(len(scores_only), 1)
    format_valid = sum(
        1 for r in results
        if (r.get("format_valid", False) if isinstance(r, dict) else False)
    )

    # Compute avg judge preference and clean bonus stats
    preferences = [r.get("judge_preference", 0) for r in results if isinstance(r, dict) and r.get("format_valid", False)]
    n_pos = sum(1 for p in preferences if p > 0)
    n_zero = sum(1 for p in preferences if p == 0)
    n_neg = sum(1 for p in preferences if p < 0)
    clean_bonuses = [r.get("clean_bonus", 0.0) for r in results if isinstance(r, dict) and r.get("format_valid", False)]
    avg_clean = sum(clean_bonuses) / max(len(clean_bonuses), 1)

    print(
        f"[BatchReward v11] samples={batch_size}, format_valid={format_valid}, "
        f"pairwise={len(preferences)} (+{n_pos}/0{n_zero}/-{n_neg}), "
        f"avg_score={avg_score:.4f}, avg_clean={avg_clean:.4f}"
    )

    # DEBUG: Print first few invalid samples to diagnose format issues
    if format_valid < batch_size:
        n_debug = min(3, batch_size - format_valid)
        debug_count = 0
        for i, r in enumerate(results):
            is_valid = r.get("format_valid", False) if isinstance(r, dict) else False
            if not is_valid:
                if debug_count < n_debug:
                    print(f"\n[DEBUG] Sample {i} format_invalid:")
                    print(f"  Output (first 500 chars): {solution_strs[i][:500]}")
                    print(f"  Output (last 200 chars): ...{solution_strs[i][-200:]}")
                    print(f"  card_pool size: {len(json.loads(ground_truths[i])['card_pool']) if isinstance(ground_truths[i], str) else len(ground_truths[i].get('card_pool', []))}")
                    debug_count += 1

    return results


def _batch_no_llm(
    solution_strs: List[str],
    ground_truths: List[Any],
    extra_infos: List[Dict],
) -> List[dict]:
    """Fallback: per-sample compute without LLM (only deterministic scoring)."""
    results = []
    for i in range(len(solution_strs)):
        gt = ground_truths[i]
        if isinstance(gt, str):
            gt = json.loads(gt)
        result = _compute_base_score(solution_strs[i], gt)
        # Strip non-scalar fields for veRL numpy compatibility
        result.pop("rank_list", None)
        result.pop("base_score", None)
        results.append(result)
    return results


def _compute_base_score(solution_str: str, gt: Dict) -> Dict[str, Any]:
    """确定性基础评分 (无 LLM), 用于 pairwise 比较的基础.

    返回:
        Dict with keys: format_valid, rank_list, base_score, rule_score,
        format_penalty, clean_bonus, and all sub-metrics
    """
    card_pool = gt.get("card_pool", [])
    positive_keys = gt.get("positive_keys", []) or gt.get("positive_actions", [])
    negative_keys = gt.get("negative_keys", [])
    ctx = gt.get("context", {})
    ctx["card_pool"] = card_pool
    ctx["positive_actions"] = positive_keys
    ctx["rule_signals"] = gt.get("rule_signals", {})

    # 1. 解析 + 格式检查
    valid, rank_list = format_gate(solution_str, card_pool)
    issues = _detect_format_issues(solution_str, card_pool, rank_list)
    format_penalty = _compute_format_penalty(issues) if valid else 0.0

    # 2. 确定性 rule score (无 LLM)
    rule_result = _rule_based_score(rank_list, positive_keys, negative_keys, ctx)
    rule_score = rule_result["score"]

    # 3. Clean bonus
    import re
    stripped_raw = solution_str.strip()
    stripped_raw = re.sub(r'<think>.*?</think>', '', stripped_raw, flags=re.DOTALL).strip()
    is_clean = stripped_raw.startswith('[') and stripped_raw.endswith(']')
    clean_bonus = 0.1 if is_clean else 0.0

    # 4. 非法输出: progressive invalid (v10)
    if not valid:
        n_valid = len(rank_list)
        if n_valid > 0:
            invalid_score = max(0.0, 0.25 - 0.1 * (3 - n_valid))
        else:
            invalid_score = -0.3
        # Return SAME keys as valid case (required by numpy array creation)
        return {
            "format_valid": False,
            "rank_list": rank_list,
            "base_score": invalid_score,
            "score": round(invalid_score, 4),
            "format_penalty": round(invalid_score, 4),
            "clean_bonus": 0.0,
            "has_markdown": False,
            "n_hallucinated": 0,
            "n_missing": 0,
            "rule_score": 0.0,
            "judge_preference": 0,
            "rule_neg_position": 0.0,
            "rule_pos_position": 0.0,
        }

    # 5. 合法输出: base_score = rule_score (确定性部分)
    # 预计算 deterministic score (无 pairwise, 供 _batch_no_llm 使用)
    det_score = round(max(0.0, min(1.0, rule_score + format_penalty + clean_bonus)), 4)
    return {
        "format_valid": True,
        "rank_list": rank_list,
        "base_score": rule_score,
        "score": det_score,  # deterministic fallback (overwritten by _batch_with_llm after pairwise)
        "format_penalty": round(format_penalty, 4),
        "clean_bonus": round(clean_bonus, 4),
        "has_markdown": issues.get("has_markdown", False),
        "n_hallucinated": issues.get("n_hallucinated", 0),
        "n_missing": issues.get("n_missing", 0),
        "rule_score": round(rule_score, 4),
        "judge_preference": 0,
        "rule_neg_position": rule_result.get("neg_avg_position", 0.0),
        "rule_pos_position": rule_result.get("pos_avg_position", 0.0),
    }


def _detect_groups(ground_truths: List[Any]) -> List[List[int]]:
    """按 ground_truth 分组 (同一 prompt 的 rollouts 共享 ground_truth).

    Returns:
        List[List[int]]: 每组包含的 sample 索引
    """
    groups = []
    current_group = [0]

    for i in range(1, len(ground_truths)):
        gt_prev = ground_truths[i - 1]
        gt_curr = ground_truths[i]
        # 比较 JSON 序列化后的字符串判断是否同一 prompt
        s_prev = json.dumps(gt_prev, sort_keys=True) if isinstance(gt_prev, dict) else str(gt_prev)
        s_curr = json.dumps(gt_curr, sort_keys=True) if isinstance(gt_curr, dict) else str(gt_curr)

        if s_prev == s_curr:
            current_group.append(i)
        else:
            groups.append(current_group)
            current_group = [i]

    groups.append(current_group)
    return groups


def _batch_with_llm(
    solution_strs: List[str],
    ground_truths: List[Any],
    extra_infos: List[Dict],
) -> List[dict]:
    """Batch mode with Pairwise Judge (v11).

    v11 公式:
      score = base_score + α × judge_preference + format_penalty + clean_bonus

      - base_score: rule_score (确定性, 无 LLM)
      - judge_preference: pairwise 比较结果 ∈ {-1, 0, +1}
      - α = 0.3: pairwise 权重
      - format_penalty: markdown -0.1, extra_text -0.05
      - clean_bonus: 纯 JSON +0.1
      - invalid: progressive penalty (-0.3 ~ 0.15)

    与 v10 (absolute judge) 的区别:
      - v10: judge 给绝对分 [-1, +1], 集中在 [-0.5, -0.7], 无区分度
      - v11: judge 给相对比较 {-1, 0, +1}, 强制离散输出, 天然有方差
    """
    batch_size = len(solution_strs)

    # ── Phase 1: 确定性基础评分 (无 LLM) ──
    results = []
    for i in range(batch_size):
        gt = ground_truths[i]
        if isinstance(gt, str):
            gt = json.loads(gt)
        result = _compute_base_score(solution_strs[i], gt)
        results.append(result)

    # ── Phase 2: 按 prompt 分组 ──
    groups = _detect_groups(ground_truths)

    # ── Phase 3: 组内 pairwise 比较 (异步) ──
    from .knowledge_judge import async_pairwise_compare

    pairs = []
    pair_map = []  # (group_idx, member_idx_in_group)

    for group_indices in groups:
        # 收集组内合法 rollout
        valid_indices = [
            i for i in group_indices if results[i]["format_valid"]
        ]

        if len(valid_indices) <= 1:
            # 组内只有 0 或 1 个合法 rollout, 无法比较
            for i in valid_indices:
                results[i]["judge_preference"] = 0
            continue

        # 按 base_score 排序, 找中位数作为锚点
        sorted_valid = sorted(valid_indices, key=lambda i: results[i]["base_score"])
        anchor_idx = sorted_valid[len(sorted_valid) // 2]
        anchor = results[anchor_idx]

        # 构建 pairwise 比较对 (每个 vs 锚点)
        gt = ground_truths[anchor_idx]
        if isinstance(gt, str):
            gt = json.loads(gt)
        pos_keys = gt.get("positive_keys", []) or gt.get("positive_actions", [])
        neg_keys = gt.get("negative_keys", [])
        ctx = gt.get("context", {})
        ctx["card_pool"] = gt.get("card_pool", [])
        ctx["positive_actions"] = pos_keys
        ctx["rule_signals"] = gt.get("rule_signals", {})

        for i in sorted_valid:
            if i == anchor_idx:
                results[i]["judge_preference"] = 0
                continue
            member = results[i]
            pairs.append((
                member["rank_list"],
                anchor["rank_list"],
                ctx,
                pos_keys,
                neg_keys,
            ))
            pair_map.append(i)

    # 异步执行所有 pairwise 比较
    if pairs:
        # Ray workers may have a running event loop; use thread pool if needed
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                preferences = executor.submit(
                    asyncio.run, async_pairwise_compare(pairs, max_concurrent=15)
                ).result()
        else:
            preferences = asyncio.run(async_pairwise_compare(pairs, max_concurrent=15))

        for idx, pref in zip(pair_map, preferences):
            results[idx]["judge_preference"] = pref

    # ── Phase 4: 合成最终 score ──
    ALPHA = 0.3  # pairwise preference 权重
    for r in results:
        if r["format_valid"]:
            score = (
                r["base_score"]
                + ALPHA * r["judge_preference"]
                + r["format_penalty"]
                + r["clean_bonus"]
            )
            r["score"] = round(max(0.0, min(1.0, score)), 4)
        # invalid 的 score 已在 _compute_base_score 中设置

    # ── Phase 5: 清理非标量字段 (veRL 需要所有字段可转换为 numpy array) ──
    for r in results:
        r.pop("rank_list", None)
        r.pop("base_score", None)

    return results
