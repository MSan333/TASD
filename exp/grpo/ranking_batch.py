"""Batch reward function for GRPO ranking task.

Called by: verl/utils/reward_score/feedback/__init__.py (compute_score_batch)
Interface: compute_score(data_sources, solution_strs, ground_truths, extra_infos) -> List[dict]

Returns List[dict] with "score" + sub-metrics (intent_alignment, strategy_compliance, etc.)
so BatchRewardManager populates reward_extra_info → SwanLab charts + console logs.

v5: Unified formula — all cases use the same scoring formula:
    final_score = α × rule_score + (1-α) × judge_score + format_penalty

优化: 同一 sample 的 K 个 rollout 共享 TipBank 检索和意图识别,
只对各自的 rank_list 做 Knowledge Judge 打分.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import os
from typing import Any, Dict, List, Optional

from .reward_fn import compute_reward, format_check

logger = logging.getLogger(__name__)


def compute_score(
    data_sources: List[str],
    solution_strs: List[str],
    ground_truths: List[Any],
    extra_infos: Optional[List[Dict]] = None,
    **kwargs,
) -> List:
    """Batch compute_score compatible with veRL BatchRewardManager.verify().

    Returns List[dict] with "score" + sub-metrics (intent_alignment, etc.)
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
    print(
        f"[BatchReward] samples={batch_size}, format_valid={format_valid}, "
        f"avg_score={avg_score:.4f}"
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
    """Fallback: per-sample compute without LLM (only format gate)."""
    results = []
    for i in range(len(solution_strs)):
        gt = ground_truths[i]
        if isinstance(gt, str):
            gt = json.loads(gt)
        result = compute_reward(solution_strs[i], gt, extra_infos[i])
        results.append({
            "score": result["score"],
            "judge_score": result.get("judge_score", 0.0),
            "rule_score": result.get("rule_score", 0.0),
            "rule_neg_position": result.get("rule_neg_position", 0.0),
            "rule_pos_position": result.get("rule_pos_position", 0.0),
            "format_valid": result.get("format_valid", False),
            "format_penalty": result.get("format_penalty", 0.0),
            "has_markdown": result.get("has_markdown", False),
            "n_hallucinated": result.get("n_hallucinated", 0),
            "n_missing": result.get("n_missing", 0),
            "intent_alignment": 0.0,
            "strategy_compliance": 0.0,
            "result_prediction": 0.0,
            "risk_avoidance": 0.0,
        })
    return results


def _batch_with_llm(
    solution_strs: List[str],
    ground_truths: List[Any],
    extra_infos: List[Dict],
) -> List[dict]:
    """Batch mode with unified formula.

    统一公式:
      final_score = α × rule_score + (1-α) × judge_score + format_penalty

      - 有 judge: α=0.5 → final = 0.5×rule + 0.5×judge + format_penalty
      - 无 judge: α=1.0 → final = rule_score + format_penalty
      - 格式不合法: 同样算 rule_score（用已解析部分），加更重的 format_penalty
    """
    from .llm_judge import batch_knowledge_judge
    from .reward_fn import _rule_based_score

    batch_size = len(solution_strs)
    parsed_gts = []
    for gt in ground_truths:
        if isinstance(gt, str):
            gt = json.loads(gt)
        parsed_gts.append(gt)

    # Pre-parse all ranking outputs with format diagnostics
    format_results = []
    llm_indices = []
    llm_rank_lists = []
    llm_gts = []

    for i in range(batch_size):
        gt = parsed_gts[i]
        card_pool = gt.get("card_pool", [])
        fc = format_check(solution_strs[i], card_pool)
        format_results.append(fc)
        if fc["valid"]:
            llm_indices.append(i)
            llm_rank_lists.append(fc["rank_list"])
            llm_gts.append(gt)

    # Run batch LLM Judge
    llm_results: List[dict] = []
    if llm_indices:
        try:
            try:
                loop = asyncio.get_running_loop()
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        asyncio.run,
                        batch_knowledge_judge(llm_rank_lists, llm_gts),
                    )
                    llm_results = future.result(timeout=600)
            except RuntimeError:
                llm_results = asyncio.run(
                    batch_knowledge_judge(llm_rank_lists, llm_gts)
                )
        except Exception as e:
            logger.warning("Batch Knowledge Judge failed: %s", e)
            llm_results = [{"score": 0.0}] * len(llm_indices)

    # Assemble final results using unified formula
    final_results: List[dict] = []
    llm_idx = 0

    for i in range(batch_size):
        fc = format_results[i]
        gt = parsed_gts[i]

        # ── 1. Rule-based Score（所有情况都算，用已解析的部分） ──
        rank_list = fc["rank_list"]
        positive_keys = gt.get("positive_keys", []) or gt.get("positive_actions", [])
        negative_keys = gt.get("negative_keys", [])
        rule_result = _rule_based_score(
            rank_list,
            positive_keys,
            negative_keys,
            gt.get("context", {}),
        )
        rule_score = rule_result["score"]

        # ── 2. Format Gate & Penalty ──
        # v5.1 fix: format-invalid outputs get ZERO for rule/judge scores,
        # only a fixed negative penalty. This prevents the degenerate strategy
        # where model outputs 1 positive key + garbage and still gets positive reward.
        # (v3 used FORMAT_PENALTY=-2.0 which provided a clear learning signal)
        if not fc["valid"]:
            # Invalid: no rule_score, no judge_score, just penalty
            if len(rank_list) == 0:
                format_penalty = -2.0   # 完全无效 — 重罚 (v3 level)
            elif len(rank_list) < 3:
                format_penalty = -1.0   # 不足 3 张 — 中罚
            else:
                format_penalty = -0.5   # 解析出 ≥3 张但有格式问题 — 轻罚

            final_score = format_penalty  # rule_score=0, judge_score=0
            judge_score = 0.0
            judge_result = {}
            rule_score = 0.0  # Don't reward partial parses on invalid output
            rule_result = {"score": 0.0, "neg_avg_position": 0.0, "pos_avg_position": 0.0}
        else:
            format_penalty = fc["penalty"]

            # ── 3. LLM Judge Score (only for valid outputs) ──
            judge_score = 0.0
            judge_result = {}
            has_judge = False

            if llm_idx < len(llm_results):
                judge_result = llm_results[llm_idx]
                judge_score = judge_result.get("score", 0.0)
                has_judge = True
                llm_idx += 1

            # ── 4. 统一公式 (only for valid outputs) ──
            if has_judge:
                # 有 judge: 50% rule + 50% judge + format_penalty
                final_score = 0.5 * rule_score + 0.5 * judge_score + format_penalty
            else:
                # 无 judge: rule + format_penalty
                final_score = rule_score + format_penalty

        final_results.append({
            "score": round(final_score, 4),
            "judge_score": round(judge_score, 4),
            "rule_score": round(rule_score, 4),
            "rule_neg_position": rule_result.get("neg_avg_position", 0.0),
            "rule_pos_position": rule_result.get("pos_avg_position", 0.0),
            "format_valid": fc["valid"],
            "format_penalty": round(format_penalty, 4),
            "has_markdown": fc.get("has_markdown", False),
            "n_hallucinated": fc.get("n_hallucinated", 0),
            "n_missing": fc.get("n_missing", 0),
            "intent_alignment": judge_result.get("intent_alignment", 0.0),
            "strategy_compliance": judge_result.get("strategy_compliance", 0.0),
            "result_prediction": judge_result.get("result_prediction", 0.0),
            "risk_avoidance": judge_result.get("risk_avoidance", 0.0),
        })

    return final_results
