"""Batch reward function for GRPO ranking task.

Called by: verl/utils/reward_score/feedback/__init__.py (compute_score_batch)
Interface: compute_score(data_sources, solution_strs, ground_truths, extra_infos) -> List[float]

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

from .reward_fn import FORMAT_PENALTY, compute_reward, format_gate

logger = logging.getLogger(__name__)


def compute_score(
    data_sources: List[str],
    solution_strs: List[str],
    ground_truths: List[Any],
    extra_infos: Optional[List[Dict]] = None,
    **kwargs,
) -> List[float]:
    """Batch compute_score compatible with veRL BatchRewardManager.verify().

    Uses batch async LLM calls for Knowledge Judge efficiency.
    Returns List[float] as expected by BatchRewardManager.
    """
    batch_size = len(solution_strs)
    if extra_infos is None:
        extra_infos = [{}] * batch_size

    use_batch_llm = bool(os.environ.get("OPENAI_API_KEY"))

    if use_batch_llm:
        scores = _batch_with_llm(solution_strs, ground_truths, extra_infos)
    else:
        scores = _batch_no_llm(solution_strs, ground_truths, extra_infos)

    avg_score = sum(scores) / max(len(scores), 1)
    format_valid = sum(1 for s in scores if s > FORMAT_PENALTY)
    print(
        f"[BatchReward] samples={batch_size}, format_valid={format_valid}, "
        f"avg_score={avg_score:.4f}"
    )

    return scores


def _batch_no_llm(
    solution_strs: List[str],
    ground_truths: List[Any],
    extra_infos: List[Dict],
) -> List[float]:
    """Fallback: per-sample compute without LLM (only format gate)."""
    results = []
    for i in range(len(solution_strs)):
        gt = ground_truths[i]
        if isinstance(gt, str):
            gt = json.loads(gt)
        result = compute_reward(solution_strs[i], gt, extra_infos[i])
        results.append(result["score"])
    return results


def _batch_with_llm(
    solution_strs: List[str],
    ground_truths: List[Any],
    extra_infos: List[Dict],
) -> List[float]:
    """Batch mode: async LLM Judge calls for all format-valid samples."""
    from .llm_judge import batch_knowledge_judge

    batch_size = len(solution_strs)
    parsed_gts = []
    for gt in ground_truths:
        if isinstance(gt, str):
            gt = json.loads(gt)
        parsed_gts.append(gt)

    # Pre-parse all ranking outputs and identify format-valid ones
    format_results = []
    llm_indices = []
    llm_rank_lists = []
    llm_gts = []

    for i in range(batch_size):
        gt = parsed_gts[i]
        card_pool = gt.get("card_pool", [])
        valid, rank_list = format_gate(solution_strs[i], card_pool)
        format_results.append((valid, rank_list))
        if valid:
            llm_indices.append(i)
            llm_rank_lists.append(rank_list)
            llm_gts.append(gt)

    # Run batch LLM Judge
    llm_scores: List[float] = []
    if llm_indices:
        try:
            try:
                loop = asyncio.get_running_loop()
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        asyncio.run,
                        batch_knowledge_judge(llm_rank_lists, llm_gts),
                    )
                    llm_scores = future.result(timeout=600)
            except RuntimeError:
                llm_scores = asyncio.run(
                    batch_knowledge_judge(llm_rank_lists, llm_gts)
                )
        except Exception as e:
            logger.warning("Batch Knowledge Judge failed: %s", e)
            llm_scores = [0.0] * len(llm_indices)

    # Assemble final scores
    final_scores: List[float] = []
    llm_idx = 0
    for i in range(batch_size):
        valid, rank_list = format_results[i]
        if not valid:
            final_scores.append(FORMAT_PENALTY)
        else:
            if llm_idx < len(llm_scores):
                final_scores.append(round(llm_scores[llm_idx], 4))
            else:
                final_scores.append(0.0)
            llm_idx += 1

    return final_scores
