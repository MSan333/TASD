"""Batch reward function for GRPO ranking task.

Called by: verl/utils/reward_score/feedback/__init__.py (compute_score_batch)
Interface: compute_score(data_sources, solution_strs, ground_truths, extra_infos) -> List[dict]

Returns List[dict] with "score" + sub-metrics (intent_alignment, strategy_compliance, etc.)
so BatchRewardManager populates reward_extra_info → SwanLab charts + console logs.

v4: Format penalty (markdown/hallucination/missing) applied on top of judge score.

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

from .reward_fn import FORMAT_PENALTY, compute_reward, format_check

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
    format_valid = sum(1 for s in scores_only if s > FORMAT_PENALTY)
    print(
        f"[BatchReward] samples={batch_size}, format_valid={format_valid}, "
        f"avg_score={avg_score:.4f}"
    )

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
            "judge_score": 0.0,  # No LLM judge in fallback mode
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
    """Batch mode: async LLM Judge returns dicts with score + sub-metrics.

    v4: Uses format_check (instead of format_gate) to get format diagnostics
    and apply format_penalty on top of judge score.
    """
    from .llm_judge import batch_knowledge_judge

    batch_size = len(solution_strs)
    parsed_gts = []
    for gt in ground_truths:
        if isinstance(gt, str):
            gt = json.loads(gt)
        parsed_gts.append(gt)

    # Pre-parse all ranking outputs with format diagnostics
    format_results = []  # List[dict] from format_check
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

    # Run batch LLM Judge — returns List[Dict] with sub-metrics
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

    # Assemble final results as dicts with format penalty applied
    final_results: List[dict] = []
    llm_idx = 0
    for i in range(batch_size):
        fc = format_results[i]
        if not fc["valid"]:
            final_results.append({
                "score": FORMAT_PENALTY,
                "judge_score": 0.0,  # No judge for invalid format
                "format_valid": False,
                "format_penalty": 0.0,
                "has_markdown": False,
                "n_hallucinated": 0,
                "n_missing": 0,
                "intent_alignment": 0.0,
                "strategy_compliance": 0.0,
                "result_prediction": 0.0,
                "risk_avoidance": 0.0,
            })
        else:
            format_penalty = fc["penalty"]
            if llm_idx < len(llm_results):
                r = llm_results[llm_idx]
                judge_score = r.get("score", 0.0)
                final_score = judge_score + format_penalty
                final_results.append({
                    "score": round(final_score, 4),
                    "judge_score": round(judge_score, 4),
                    "format_valid": True,
                    "format_penalty": format_penalty,
                    "has_markdown": fc["has_markdown"],
                    "n_hallucinated": fc["n_hallucinated"],
                    "n_missing": fc["n_missing"],
                    "intent_alignment": r.get("intent_alignment", 0.0),
                    "strategy_compliance": r.get("strategy_compliance", 0.0),
                    "result_prediction": r.get("result_prediction", 0.0),
                    "risk_avoidance": r.get("risk_avoidance", 0.0),
                })
            else:
                # Fallback when LLM judge failed or returned fewer results
                final_results.append({
                    "score": round(format_penalty, 4),
                    "judge_score": 0.0,
                    "format_valid": True,
                    "format_penalty": format_penalty,
                    "has_markdown": fc["has_markdown"],
                    "n_hallucinated": fc["n_hallucinated"],
                    "n_missing": fc["n_missing"],
                    "intent_alignment": 0.0,
                    "strategy_compliance": 0.0,
                    "result_prediction": 0.0,
                    "risk_avoidance": 0.0,
                })
            llm_idx += 1

    return final_results
