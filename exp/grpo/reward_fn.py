"""Knowledge-Grounded Reward Function (v3).

veRL 调用链:
  feedback/__init__.py → exp/grpo/ranking.py → 本文件 compute_reward()

架构 (两层):
  L0: Format Gate — 格式校验, 不合法返回 -2.0
  L2: Knowledge-Grounded Judge — TipBank 检索 + 意图识别 + 知识对比打分

环境变量:
  TIPBANK_PATH: TipBank 数据目录 (JSON 文件夹)
  OPENAI_API_KEY / OPENAI_BASE_URL: LLM 调用配置
  JUDGE_MODEL: 打分模型 (默认 qwen-plus)
  QUERY_MODEL: 检索 query 生成模型 (默认 qwen-turbo)
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

FORMAT_PENALTY = -2.0

# ============================================================================
# 进程级单例
# ============================================================================

_TIP_INDEX = None
_LLM_CLIENT = None


def _get_tip_index():
    global _TIP_INDEX
    if _TIP_INDEX is not None:
        return _TIP_INDEX

    tipbank_path = os.environ.get("TIPBANK_PATH", "")
    if not tipbank_path or not os.path.exists(tipbank_path):
        logger.warning("TipBank not found at TIPBANK_PATH=%s, Knowledge Judge will degrade", tipbank_path)
        return None

    try:
        from .tipbank_loader import load_tipbank, TipIndex
        tips = load_tipbank(tipbank_path)
        if tips:
            _TIP_INDEX = TipIndex(tips)
            logger.info("TipBank loaded: %s", _TIP_INDEX.get_stats())
        return _TIP_INDEX
    except Exception as e:
        logger.warning("TipBank load failed: %s", e)
        return None


def _get_llm_client():
    global _LLM_CLIENT
    if _LLM_CLIENT is not None:
        return _LLM_CLIENT

    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set, Knowledge Judge unavailable")
        return None

    _LLM_CLIENT = OpenAI(api_key=api_key, base_url=base_url)
    return _LLM_CLIENT


# ============================================================================
# L0: Format Gate
# ============================================================================

def _parse_ranking(raw: str, card_pool: List[str]) -> List[str]:
    """解析模型输出为 rank_list, 去重并过滤非法卡片."""
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s).strip()
    if "</think>" in s:
        s = s.split("</think>")[-1].strip()
    try:
        arr = json.loads(s)
        if isinstance(arr, list):
            pool_set = set(card_pool)
            seen: set = set()
            result = []
            for x in arr:
                if isinstance(x, str) and x in pool_set and x not in seen:
                    result.append(x)
                    seen.add(x)
            return result
    except (json.JSONDecodeError, TypeError):
        pass
    return []


def format_gate(raw: str, card_pool: List[str]) -> Tuple[bool, List[str]]:
    """L0 格式校验: 合法排序至少 3 张有效卡片."""
    rank_list = _parse_ranking(raw, card_pool)
    if len(rank_list) < 3:
        return False, []
    return True, rank_list


# ============================================================================
# TipBank 检索
# ============================================================================

def _retrieve_tips(ctx: Dict, card_pool: List[str], client, model: str) -> list:
    """为 reward 评分检索相关策略."""
    tip_index = _get_tip_index()
    if not tip_index:
        return []

    try:
        from .tipbank_loader import generate_search_queries
        queries = generate_search_queries(
            client, model, ctx, card_pool,
            extra_context=f"商家随后执行了: {ctx.get('positive_actions', [])}",
        )
        tips = tip_index.search(
            queries=queries,
            top_k=15,
            venture=ctx.get("venture", ""),
            action_names=card_pool,
        )
        return tips
    except Exception as e:
        logger.warning("TipBank retrieval failed: %s", e)
        return []


# ============================================================================
# Main Entry: compute_reward
# ============================================================================

def compute_reward(
    solution_str: str,
    ground_truth: Dict[str, Any],
    extra_info: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Knowledge-Grounded reward computation.

    两层架构:
      L0: Format Gate → 不合法返回 -2.0
      L2: Knowledge Judge → TipBank 检索 + 意图识别 + 知识对比打分

    Returns:
        {"score": float, "format_valid": bool, "feedback": str, ...}
    """
    card_pool = ground_truth.get("card_pool", [])
    positive_keys = ground_truth.get("positive_keys", [])
    negative_keys = ground_truth.get("negative_keys", [])
    ctx = ground_truth.get("context", {})
    ctx["card_pool"] = card_pool
    ctx["positive_actions"] = positive_keys

    # L0: Format Gate
    valid, rank_list = format_gate(solution_str, card_pool)
    if not valid:
        return {
            "score": FORMAT_PENALTY,
            "format_valid": False,
            "feedback": "format_invalid",
            "rank_list": [],
        }

    # L2: Knowledge-Grounded Judge
    client = _get_llm_client()
    if not client:
        return {
            "score": 0.0,
            "format_valid": True,
            "feedback": "llm_unavailable",
            "rank_list": rank_list,
        }

    judge_model = os.environ.get("JUDGE_MODEL", "qwen-plus")
    query_model = os.environ.get("QUERY_MODEL", "qwen-turbo")

    # Step 1: TipBank Retrieval
    tips = _retrieve_tips(ctx, card_pool, client, query_model)

    # Step 2: Intent Recognition
    from .intent_recognizer import recognize_intent
    intent = recognize_intent(ctx, client, judge_model)

    # Step 3: Knowledge Judge
    from .knowledge_judge import knowledge_judge_single
    judge_result = knowledge_judge_single(
        rank_list, ctx, tips, intent,
        positive_keys, negative_keys,
        client, judge_model,
    )

    score = judge_result.get("score", 0.0)

    return {
        "score": round(score, 4),
        "format_valid": True,
        "intent_alignment": judge_result.get("intent_alignment"),
        "strategy_compliance": judge_result.get("strategy_compliance"),
        "result_prediction": judge_result.get("result_prediction"),
        "risk_avoidance": judge_result.get("risk_avoidance"),
        "rank_list": rank_list,
        "feedback": judge_result.get("reasoning", ""),
    }
