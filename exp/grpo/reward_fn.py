"""Knowledge-Grounded Reward Function (v4 — with format penalties).

veRL 调用链:
  feedback/__init__.py → exp/grpo/ranking.py → 本文件 compute_reward()

架构 (三层):
  L0: Format Gate + Penalty — 格式校验 + Markdown/幻觉/缺失惩罚
  L1: Format Check — 返回详细格式诊断信息
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
_THINK_END = "\x3c/think\x3e"

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
# L0: Format Gate + Penalty
# ============================================================================

def _parse_ranking(raw: str, card_pool: List[str]) -> List[str]:
    """解析模型输出为 rank_list, 去重并过滤非法卡片."""
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s).strip()
    if _THINK_END in s:
        s = s.split(_THINK_END)[-1].strip()
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
# L1: Format Check — 详细格式诊断 + 惩罚计算
# ============================================================================

def _detect_format_issues(raw: str, card_pool: List[str], rank_list: List[str]) -> Dict[str, Any]:
    """检测原始输出中的格式问题（在 parse 剥离之前检查）.

    检测项:
      - has_markdown: 输出中是否包含 ``` 代码块包裹
      - n_hallucinated: 输出了多少不在 card_pool 中的 key
      - n_missing: card_pool 中有多少 key 未被输出
    """
    has_markdown = bool(
        re.search(r'```\w*\s', raw) or re.search(r'\s```', raw)
    )
    stripped = raw.strip()
    if stripped.startswith('```') or stripped.endswith('```'):
        has_markdown = True

    # Re-parse to see what model actually output (before filtering)
    pool_set = set(card_pool)
    s = stripped
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s).strip()
    if _THINK_END in s:
        s = s.split(_THINK_END)[-1].strip()

    hallucinated_keys = []
    try:
        arr = json.loads(s)
        if isinstance(arr, list):
            seen_hallucinated = set()
            for item in arr:
                if isinstance(item, str) and item not in pool_set and item not in seen_hallucinated:
                    hallucinated_keys.append(item)
                    seen_hallucinated.add(item)
    except (json.JSONDecodeError, TypeError):
        pass

    n_valid = len(rank_list)
    n_pool = len(card_pool)
    n_missing = max(0, n_pool - n_valid)

    return {
        "has_markdown": has_markdown,
        "n_hallucinated": len(hallucinated_keys),
        "hallucinated_keys": hallucinated_keys,
        "n_missing": n_missing,
    }


def _compute_format_penalty(issues: Dict) -> float:
    """计算格式惩罚（叠加到 judge score 上）.

    设计原则 (GRPO 友好):
      - judge 信号 (range=2.0) >> 格式惩罚 (max=0.8)
      - 格式惩罚占 judge 的 ~40%，作为辅助信号
      - 同组 rollout 共享的格式问题 (如 markdown) 给低惩罚
      - 个体差异的格式问题 (如幻觉) 给中等惩罚
      - 格式无效的 FORMAT_PENALTY=-2.0 负责兜底

    Penalty structure:
      - Markdown wrapping:     -0.2   (强先验, 同组共享, 低惩罚)
      - Hallucinated keys:     -0.3 each, max -0.5  (个体差异, 中惩罚)
      - Missing pool cards:    -0.05 each, max -0.3  (易恢复, 低惩罚)
    总惩罚上限: -0.8
    """
    p = 0.0
    if issues["has_markdown"]:
        p -= 0.2
    n_h = issues["n_hallucinated"]
    if n_h > 0:
        p -= min(0.5, 0.3 * n_h)
    n_m = issues["n_missing"]
    if n_m > 0:
        p -= min(0.3, 0.05 * n_m)
    # Cap total format penalty to keep judge as dominant signal
    p = max(p, -0.8)
    return round(p, 4)


def format_check(raw: str, card_pool: List[str]) -> Dict[str, Any]:
    """Format validation with detailed diagnostics and penalty.

    Returns dict: valid, rank_list, penalty, has_markdown,
    n_hallucinated, n_missing, hallucinated_keys.
    """
    valid, rank_list = format_gate(raw, card_pool)
    issues = _detect_format_issues(raw, card_pool, rank_list)
    penalty = _compute_format_penalty(issues) if valid else 0.0

    return {
        "valid": valid,
        "rank_list": rank_list,
        "penalty": penalty,
        "has_markdown": issues["has_markdown"],
        "n_hallucinated": issues["n_hallucinated"],
        "n_missing": issues["n_missing"],
        "hallucinated_keys": issues.get("hallucinated_keys", []),
    }


# ============================================================================
# Main Entry: compute_reward
# ============================================================================

def compute_reward(
    solution_str: str,
    ground_truth: Dict[str, Any],
    extra_info: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Knowledge-Grounded reward computation (v4).

    三层架构:
      L0: Format Gate → 不合法返回 -2.0
      L0b: Format Penalty → Markdown/幻觉/缺失惩罚叠加到 judge score
      L2: Knowledge Judge → TipBank 检索 + 意图识别 + 知识对比打分

    Returns:
        {"score": float, "format_valid": bool, "format_penalty": float,
         "has_markdown": bool, "n_hallucinated": int, "n_missing": int,
         "feedback": str, "rank_list": [...], ...}
    """
    card_pool = ground_truth.get("card_pool", [])
    positive_keys = ground_truth.get("positive_keys", [])
    negative_keys = ground_truth.get("negative_keys", [])
    ctx = ground_truth.get("context", {})
    ctx["card_pool"] = card_pool
    ctx["positive_actions"] = positive_keys

    # L0+L1: Format Gate + Check
    fc = format_check(solution_str, card_pool)
    if not fc["valid"]:
        return {
            "score": FORMAT_PENALTY,
            "format_valid": False,
            "format_penalty": 0.0,
            "has_markdown": False,
            "n_hallucinated": 0,
            "n_missing": 0,
            "feedback": "format_invalid",
            "rank_list": [],
        }

    rank_list = fc["rank_list"]
    format_penalty = fc["penalty"]
    has_markdown = fc["has_markdown"]
    n_hallucinated = fc["n_hallucinated"]
    n_missing = fc["n_missing"]

    # Log format issues (sample-level, rate-limited by caller)
    if format_penalty < 0:
        parts = []
        if has_markdown:
            parts.append("markdown")
        if n_hallucinated:
            parts.append(f"hallucinated={n_hallucinated}")
        if n_missing:
            parts.append(f"missing={n_missing}")
        logger.debug("[FormatPenalty] %s → %.2f", ", ".join(parts), format_penalty)

    # L2: Knowledge-Grounded Judge
    client = _get_llm_client()
    if not client:
        return {
            "score": format_penalty,
            "format_valid": True,
            "format_penalty": format_penalty,
            "has_markdown": has_markdown,
            "n_hallucinated": n_hallucinated,
            "n_missing": n_missing,
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

    judge_score = judge_result.get("score", 0.0)
    final_score = judge_score + format_penalty

    return {
        "score": round(final_score, 4),
        "format_valid": True,
        "format_penalty": format_penalty,
        "has_markdown": has_markdown,
        "n_hallucinated": n_hallucinated,
        "n_missing": n_missing,
        "intent_alignment": judge_result.get("intent_alignment", 0.0),
        "strategy_compliance": judge_result.get("strategy_compliance", 0.0),
        "result_prediction": judge_result.get("result_prediction", 0.0),
        "risk_avoidance": judge_result.get("risk_avoidance", 0.0),
        "judge_score": round(judge_score, 4),
        "rank_list": rank_list,
        "feedback": judge_result.get("reasoning", ""),
    }


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
