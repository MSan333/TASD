"""Batch reward function for GRPO ranking task.

Designed for use with veRL's BatchRewardManager:
  - compute_score receives lists (data_sources, solution_strs, ground_truths, extra_infos)
  - Calls reward_v2 for eval+rule scores per sample
  - Calls batch_llm_judge in async parallel for LLM scores
  - Merges them with proper weights
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def compute_score(
    data_sources: List[str],
    solution_strs: List[str],
    ground_truths: List[Any],
    extra_infos: Optional[List[Dict]] = None,
    **kwargs,
) -> List[Dict[str, Any]]:
    """Batch compute_score compatible with BatchRewardManager.verify().

    Steps:
      1. Compute eval_score + rule_score per sample (fast, no I/O)
      2. Collect samples eligible for LLM Judge
      3. Call batch_llm_judge in one async batch
      4. Merge LLM scores back into final results
    """
    from .reward_v2 import (
        FORMAT_PENALTY,
        REWARD_CLIP,
        WEIGHTS,
        WEIGHTS_NO_LLM,
        compute_eval_score,
        compute_rule_score,
        format_gate,
    )

    batch_size = len(solution_strs)
    if extra_infos is None:
        extra_infos = [{}] * batch_size

    # --- Phase 1: Per-sample eval + rule (CPU-only, fast) ---
    results: List[Dict[str, Any]] = []
    llm_eligible_indices: List[int] = []
    llm_rank_lists: List[List[str]] = []
    llm_contexts: List[Dict[str, Any]] = []
    llm_positive_keys: List[List[str]] = []
    llm_negative_keys: List[List[str]] = []

    for i in range(batch_size):
        gt = ground_truths[i]
        if isinstance(gt, str):
            gt = json.loads(gt)

        card_pool = gt.get("card_pool", [])
        positive_keys = gt.get("positive_keys", [])
        negative_keys = gt.get("negative_keys", [])
        positive_weights = gt.get("positive_weights", {})
        negative_weights = gt.get("negative_weights", {})
        rule_signals = gt.get("rule_signals", {})

        # L0: Format gate
        valid, rank_list = format_gate(solution_strs[i], card_pool)
        if not valid:
            results.append({
                "score": FORMAT_PENALTY,
                "format_valid": False,
                "eval_score": 0.0,
                "rule_score": 0.0,
                "llm_score": None,
                "rank_list": [],
            })
            continue

        # L1: eval_score
        eval_score = compute_eval_score(
            rank_list, positive_keys, negative_keys,
            positive_weights, negative_weights,
        )

        # L2: rule_score
        rule_score = compute_rule_score(rank_list, rule_signals)

        # Placeholder for llm_score (will fill later)
        results.append({
            "eval_score": eval_score,
            "rule_score": rule_score,
            "llm_score": None,
            "format_valid": True,
            "rank_list": rank_list,
            "score": 0.0,  # will be computed after LLM
        })

        # Check if LLM Judge is eligible
        context = gt.get("context")
        if context and os.environ.get("OPENAI_API_KEY"):
            if isinstance(context, str):
                context = json.loads(context)
            if "card_pool" not in context:
                context["card_pool"] = card_pool
            llm_eligible_indices.append(i)
            llm_rank_lists.append(rank_list)
            llm_contexts.append(context)
            llm_positive_keys.append(positive_keys)
            llm_negative_keys.append(negative_keys)

    # --- Phase 2: Batch LLM Judge (async parallel) ---
    llm_scores: List[float] = []
    if llm_eligible_indices:
        try:
            from .llm_judge import batch_llm_judge

            try:
                loop = asyncio.get_running_loop()
                # If already in an event loop, create a new thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        asyncio.run,
                        batch_llm_judge(
                            llm_rank_lists,
                            llm_contexts,
                            llm_positive_keys,
                            llm_negative_keys,
                            max_concurrent=10,
                        ),
                    )
                    llm_scores = future.result(timeout=600)
            except RuntimeError:
                # No running event loop
                llm_scores = asyncio.run(
                    batch_llm_judge(
                        llm_rank_lists,
                        llm_contexts,
                        llm_positive_keys,
                        llm_negative_keys,
                        max_concurrent=10,
                    )
                )
        except Exception as e:
            logger.warning("Batch LLM Judge failed: %s", e)
            llm_scores = [0.0] * len(llm_eligible_indices)

    # --- Phase 3: Merge LLM scores and compute final ---
    for batch_idx, llm_idx in enumerate(llm_eligible_indices):
        if batch_idx < len(llm_scores):
            results[llm_idx]["llm_score"] = llm_scores[batch_idx]

    # Compute final weighted score for all valid results
    for result in results:
        if not result["format_valid"]:
            continue

        eval_score = result["eval_score"]
        rule_score = result["rule_score"]
        llm_score = result["llm_score"]

        if llm_score is not None:
            total = (WEIGHTS["eval"] * eval_score +
                     WEIGHTS["rule"] * rule_score +
                     WEIGHTS["llm"] * llm_score)
        else:
            total = (WEIGHTS_NO_LLM["eval"] * eval_score +
                     WEIGHTS_NO_LLM["rule"] * rule_score)

        result["score"] = round(max(-REWARD_CLIP, min(REWARD_CLIP, total)), 4)

    # Print reward summary (visible in training logs)
    avg_score = sum(r["score"] for r in results) / max(len(results), 1)
    format_valid_count = sum(1 for r in results if r.get("format_valid"))
    llm_scores_summary = [r.get("llm_score") for r in results if r.get("llm_score") is not None]
    avg_llm = sum(llm_scores_summary) / max(len(llm_scores_summary), 1) if llm_scores_summary else 0.0

    print(
        f"[BatchReward] samples={batch_size}, format_valid={format_valid_count}, "
        f"llm_judged={len(llm_eligible_indices)}, "
        f"avg_score={avg_score:.4f}, avg_llm={avg_llm:.4f}"
    )

    # Return only the float scores (batch.py expects List[float])
    return [r["score"] for r in results]
