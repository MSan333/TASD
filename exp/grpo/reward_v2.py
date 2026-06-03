"""GRPO v2 Reward Function: 三层混合架构.

架构:
  L0: 格式门槛 (硬约束, -2.0)
  L1: eval_score (45%) — 直接对齐评估器指标
  L2: rule_score (20%) — 确定性规则约束
  L3: llm_score (35%) — LLM Judge 排序合理性 (可选, batch 调用)

veRL 接口: compute_score(solution_str, ground_truth, extra_info)
"""

from __future__ import annotations

import json
import logging
import math
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

FORMAT_PENALTY = -2.0
REWARD_CLIP = 2.0

# 权重配置
WEIGHTS = {
    "eval": 0.45,
    "rule": 0.20,
    "llm": 0.35,
}

# 无 LLM 时的 fallback 权重
WEIGHTS_NO_LLM = {
    "eval": 0.60,
    "rule": 0.40,
}


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _rank_of(target: str, ranking: List[str]) -> Optional[int]:
    """返回 target 在 ranking 中的 1-based 位置."""
    for i, k in enumerate(ranking, start=1):
        if k == target:
            return i
    return None


def _parse_ranking(raw: str, card_pool: List[str]) -> List[str]:
    """解析 LLM 输出为排序列表."""
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s).strip()
    try:
        arr = json.loads(s)
        if isinstance(arr, list):
            return [x for x in arr if isinstance(x, str)]
    except Exception:
        pass
    keys: List[str] = []
    seen = set()
    for m in re.findall(r'"([^"]+)"', s):
        if m in card_pool and m not in seen:
            keys.append(m)
            seen.add(m)
    return keys


# ---------------------------------------------------------------------------
# L0: 格式门槛
# ---------------------------------------------------------------------------

def format_gate(raw_output: str, card_pool: List[str]) -> Tuple[bool, List[str]]:
    """格式校验. 不合法返回 (False, [])."""
    rank_list = _parse_ranking(raw_output, card_pool)
    if not rank_list or len(rank_list) < 3:
        return False, []

    pool_set = set(card_pool)
    seen = set()
    valid = []
    for k in rank_list:
        if k in pool_set and k not in seen:
            valid.append(k)
            seen.add(k)

    if len(valid) < 3:
        return False, []
    return True, valid


# ---------------------------------------------------------------------------
# L1: 评估指标得分 (eval_score)
# ---------------------------------------------------------------------------

def compute_eval_score(
    rank_list: List[str],
    positive_keys: List[str],
    negative_keys: List[str],
    positive_weights: Optional[Dict[str, float]] = None,
    negative_weights: Optional[Dict[str, float]] = None,
) -> float:
    """对齐评估器公式 + 位置连续打分.

    eval_score = 0.7 × metric_score + 0.3 × position_score
    """
    top1 = set(rank_list[:1])
    top3 = set(rank_list[:3])
    pos_set = set(positive_keys)
    neg_set = set(negative_keys)

    # A. 评估指标层 (离散)
    pos_hit_1 = 1.0 if (top1 & pos_set) else 0.0
    pos_hit_3 = 1.0 if (top3 & pos_set) else 0.0
    avoid_1 = 0.0 if (top1 & neg_set) else 1.0
    avoid_3 = 0.0 if (top3 & neg_set) else 1.0

    good = bad = 0
    for pk in positive_keys:
        if pk in top1:
            good += 1
        else:
            bad += 1
    for nk in negative_keys:
        if nk not in top1:
            good += 1
        else:
            bad += 1
    good_rate_1 = good / (good + bad) if (good + bad) > 0 else 0.5

    metric_score = (pos_hit_1 * 0.30 + pos_hit_3 * 0.20 +
                    avoid_1 * 0.20 + avoid_3 * 0.10 + good_rate_1 * 0.20)

    # B. 位置连续层 (带因果权重)
    pos_w = positive_weights or {}
    neg_w = negative_weights or {}

    pos_position_score = 0.0
    for pk in positive_keys:
        w = pos_w.get(pk, 1.0)
        rank = _rank_of(pk, rank_list)
        if rank is None:
            pos_position_score += 0.0
        elif rank == 1:
            pos_position_score += 1.0 * w
        elif rank <= 3:
            pos_position_score += 0.6 * w
        elif rank <= 5:
            pos_position_score += 0.3 * w
        elif rank <= 8:
            pos_position_score += 0.1 * w

    neg_position_score = 0.0
    for nk in negative_keys:
        w = neg_w.get(nk, 1.0)
        rank = _rank_of(nk, rank_list)
        if rank is None or rank > 8:
            neg_position_score += 1.0 * w
        elif rank > 5:
            neg_position_score += 0.6 * w
        elif rank > 3:
            neg_position_score += 0.3 * w
        elif rank > 1:
            neg_position_score += 0.1 * w
        else:
            neg_position_score += 0.0

    n_signals = len(positive_keys) + len(negative_keys)
    if n_signals > 0:
        position_score = (pos_position_score + neg_position_score) / n_signals
    else:
        position_score = 0.5

    return 0.7 * metric_score + 0.3 * position_score


# ---------------------------------------------------------------------------
# L2: 规则得分 (rule_score)
# ---------------------------------------------------------------------------

def compute_rule_score(rank_list: List[str], rule_signals: Dict[str, Any]) -> float:
    """确定性规则打分. 返回 [-1.0, 1.0]."""
    if not rule_signals:
        return 0.0

    scores = []

    # 规则1: OOBA
    topup_in_top3 = [c for c in rank_list[:3] if "topup" in c]
    if rule_signals.get("has_ooba"):
        scores.append(0.5 if topup_in_top3 else -0.5)
    else:
        if topup_in_top3:
            scores.append(-0.3 * len(topup_in_top3))
        else:
            scores.append(0.2)

    # 规则2: OOBU
    for plan_type in rule_signals.get("oobu_plans", []):
        budget_card = f"{plan_type}_budget_increase"
        if budget_card in rank_list[:5]:
            scores.append(0.5)
        elif budget_card in rank_list:
            scores.append(-0.2)

    # 规则3: 计划不存在
    missing_plans = list(rule_signals.get("missing_plans", []))
    for card in rank_list[:5]:
        for missing in missing_plans:
            if card.startswith(str(missing) + "_") and "creation" not in card:
                scores.append(-0.5)
                break

    # 规则4: budget_vs_troi 排序
    for plan_type, info in rule_signals.get("budget_vs_troi", {}).items():
        if not isinstance(info, dict) or not info.get("expect"):
            continue
        budget_card = f"{plan_type}_budget_increase"
        troi_card = f"{plan_type}_troi"
        b_rank = _rank_of(budget_card, rank_list)
        t_rank = _rank_of(troi_card, rank_list)
        if b_rank is not None and t_rank is not None:
            if info["expect"] == "budget_first":
                scores.append(0.3 if b_rank < t_rank else -0.3)
            elif info["expect"] == "troi_first":
                scores.append(0.3 if t_rank < b_rank else -0.3)

    if not scores:
        return 0.0

    raw = sum(scores) / len(scores)
    return max(-1.0, min(1.0, raw))


# ---------------------------------------------------------------------------
# 主入口: veRL compute_score 接口
# ---------------------------------------------------------------------------

def compute_score(
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[Dict] = None,
) -> Dict[str, Any]:
    """veRL reward interface.

    Args:
        solution_str: 模型的原始文本输出
        ground_truth: reward_model.ground_truth 字段
        extra_info: 额外元信息

    Returns:
        {"score": float, ...breakdown...}
    """
    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)

    card_pool = ground_truth.get("card_pool", [])
    positive_keys = ground_truth.get("positive_keys", [])
    negative_keys = ground_truth.get("negative_keys", [])
    positive_weights = ground_truth.get("positive_weights", {})
    negative_weights = ground_truth.get("negative_weights", {})
    rule_signals = ground_truth.get("rule_signals", {})

    # L0: 格式门槛
    valid, rank_list = format_gate(solution_str, card_pool)
    if not valid:
        return {
            "score": FORMAT_PENALTY,
            "format_valid": False,
            "eval_score": 0.0,
            "rule_score": 0.0,
            "llm_score": None,
            "rank_list": [],
        }

    # L1: eval_score
    eval_score = compute_eval_score(
        rank_list, positive_keys, negative_keys,
        positive_weights, negative_weights,
    )

    # L2: rule_score
    rule_score = compute_rule_score(rank_list, rule_signals)

    # L3: llm_score (由外部 batch 填充, 此处默认 None)
    llm_score = None
    if extra_info and "llm_score" in extra_info:
        llm_score = extra_info["llm_score"]

    # 加权
    if llm_score is not None:
        total = (WEIGHTS["eval"] * eval_score +
                 WEIGHTS["rule"] * rule_score +
                 WEIGHTS["llm"] * llm_score)
    else:
        total = (WEIGHTS_NO_LLM["eval"] * eval_score +
                 WEIGHTS_NO_LLM["rule"] * rule_score)

    total = max(-REWARD_CLIP, min(REWARD_CLIP, total))

    return {
        "score": round(total, 4),
        "format_valid": True,
        "eval_score": round(eval_score, 4),
        "rule_score": round(rule_score, 4),
        "llm_score": round(llm_score, 4) if llm_score is not None else None,
        "rank_list": rank_list,
    }
