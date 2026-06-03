"""veRL reward adapter for GRPO ranking task.

Dispatches to reward_v2 (ranking_v2) or legacy grpo_reward (ranking) based on
the ``style`` field in reward_model.
"""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def compute_score(
    solution_str: str,
    ground_truth: Any,
    extra_info: dict = None,
) -> dict:
    """Compute ranking reward compatible with veRL's reward interface."""
    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)

    is_v2 = "positive_keys" in ground_truth or "rule_signals" in ground_truth
    if is_v2:
        from .reward_v2 import compute_score as compute_v2
        return compute_v2(solution_str, ground_truth, extra_info)

    from .grpo_reward import compute_grpo_reward

    card_pool = ground_truth.get("card_pool", [])
    clean_pos = ground_truth.get("clean_positive_keys", [])
    weak_pos = ground_truth.get("weak_positive_keys", [])
    clean_neg = ground_truth.get("clean_negative_keys", [])

    reward_weights = None
    if extra_info and "reward_weights" in extra_info:
        reward_weights = extra_info["reward_weights"]

    total_reward, breakdown = compute_grpo_reward(
        raw_output=solution_str,
        card_pool=card_pool,
        clean_positive_keys=clean_pos,
        clean_negative_keys=clean_neg,
        weak_positive_keys=weak_pos,
        weights=reward_weights,
    )

    return {
        "score": total_reward,
        "format_valid": breakdown.get("format_valid", False),
        "pos_hit": breakdown.get("pos_hit", 0.0),
        "neg_penalty": breakdown.get("neg_penalty", 0.0),
        "coverage": breakdown.get("coverage", 0.0),
        "llm_judge": breakdown.get("llm_judge"),
        "llm_triggered": breakdown.get("llm_triggered", False),
        "feedback": "",
    }
