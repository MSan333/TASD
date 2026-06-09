"""veRL single-sample reward adapter for GRPO ranking task.

Called by: verl/utils/reward_score/feedback/__init__.py
Interface: compute_score(solution_str, ground_truth, extra_info) -> dict
"""

import json
import logging
from typing import Any, Dict, Optional

from .reward_fn import compute_reward

logger = logging.getLogger(__name__)


def compute_score(
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[Dict] = None,
) -> Dict[str, Any]:
    """veRL NaiveRewardManager 单条 reward 接口."""
    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)

    return compute_reward(solution_str, ground_truth, extra_info)
