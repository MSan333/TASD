"""Knowledge-Grounded Reward Function (v5 — unified formula).

veRL 调用链:
  feedback/__init__.py → exp/grpo/ranking.py → 本文件 compute_reward()

统一公式:
  final_score = α × rule_score + (1-α) × judge_score + format_penalty

  - 有 LLM Judge: α=0.5 → final = 0.5×rule + 0.5×judge + format_penalty
  - 无 LLM Judge: α=1.0 → final = rule_score + format_penalty
  - 格式不合法:  同样算 rule_score（用已解析部分），加更重的 format_penalty

格式惩罚（较轻，辅助信号）:
  - 完全无效（0张卡）: -0.3
  - 部分有效（<3张卡）: -0.15
  - Markdown 包裹:     -0.1
  - 幻觉卡片:          -0.15/个, max -0.25
  - 遗漏卡片:          -0.025/个, max -0.15
  - 总惩罚上限:        -0.4

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
    """解析模型输出为 rank_list, 去重并过滤非法卡片.

    剥离顺序:
      1. 移除 <think>...</think> 块（包括未闭合的 ）
      2. 移除 ``` markdown 代码块包裹
      3. **提取 JSON 部分**（如果前面有思考文本）
      4. 解析 JSON 数组
      5. 过滤掉不在 card_pool 中的 key
    """
    s = raw.strip()
    # 1. 剥离 <think>...</think> 块（处理闭合和未闭合两种情况）
    s = re.sub(r'<think>.*?</think>', '', s, flags=re.DOTALL)
    s = re.sub(r'<think>.*$', '', s, flags=re.DOTALL)  # 未闭合的
    # 2. 移除 markdown 代码块
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s).strip()

    # 3. **提取 JSON 部分** — 找到第一个 [ 和最后一个 ] 之间的内容
    # 这能处理模型输出思考内容后跟 JSON 的情况
    start_idx = s.find('[')
    end_idx = s.rfind(']')
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        s = s[start_idx:end_idx + 1]

    # 4. 解析 JSON (长度限制防止超深嵌套导致 RecursionError)
    if len(s) > 10000:
        return []
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
    except (json.JSONDecodeError, TypeError, ValueError, OverflowError, RecursionError):
        pass
    return []


def format_gate(raw: str, card_pool: List[str]) -> Tuple[bool, List[str]]:
    """L0 格式校验: 合法排序至少 3 张有效卡片."""
    rank_list = _parse_ranking(raw, card_pool)
    if len(rank_list) < 3:
        return False, rank_list  # 仍然返回已解析的部分，用于渐进打分
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
    # Strip thinking blocks (same as _parse_ranking)
    s = re.sub(r'<think>.*?</think>', '', s, flags=re.DOTALL)
    s = re.sub(r'<think>.*$', '', s, flags=re.DOTALL)
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s).strip()

    # Extract JSON part (same as _parse_ranking)
    start_idx = s.find('[')
    end_idx = s.rfind(']')
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        s = s[start_idx:end_idx + 1]

    # 4. 检测幻觉 key（长度限制防止超深嵌套）
    hallucinated_keys = []
    if len(s) > 10000:
        return {
            "has_markdown": has_markdown,
            "n_hallucinated": 0,
            "hallucinated_keys": [],
            "n_missing": max(0, len(card_pool) - len(rank_list)),
        }
    try:
        arr = json.loads(s)
        if isinstance(arr, list):
            seen_hallucinated = set()
            for item in arr:
                if isinstance(item, str) and item not in pool_set and item not in seen_hallucinated:
                    hallucinated_keys.append(item)
                    seen_hallucinated.add(item)
    except (json.JSONDecodeError, TypeError, ValueError, OverflowError, RecursionError):
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
    """计算格式惩罚（叠加到最终得分上）.

    设计原则 (GRPO 友好):
      - rule/judge 信号是主信号，格式惩罚是辅助
      - 总惩罚上限 -0.4，不至于压过排序质量信号

    Penalty structure:
      - Markdown wrapping:     -0.1
      - Hallucinated keys:     -0.15 each, max -0.25
      - Missing pool cards:    -0.025 each, max -0.15
    总惩罚上限: -0.4
    """
    p = 0.0
    if issues["has_markdown"]:
        p -= 0.1
    n_h = issues["n_hallucinated"]
    if n_h > 0:
        p -= min(0.25, 0.15 * n_h)
    n_m = issues["n_missing"]
    if n_m > 0:
        p -= min(0.15, 0.025 * n_m)
    p = max(p, -0.4)
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
# L1b: Rule-based Score — 位置加权排序评分
# ============================================================================

def _rule_based_score(
    rank_list: List[str],
    positive_keys: List[str],
    negative_keys: List[str],
    ctx: Dict[str, Any],
) -> Dict[str, Any]:
    """位置加权的排序质量评分，覆盖所有位置.

    核心思路：每张卡片在每个位置都有信号，而不是只看 Top-3.
    这让 GRPO 组内有更大方差，模型能学到更细粒度的排序策略.

    评分维度:
      1. negative (60%): 负样本应在排名靠后
         - 在 Top-3 → 重罚 (越靠前罚越重)
         - 在中段   → 轻罚
         - 在后段   → 奖励
         - 未出现   → 中性 (0)
      2. positive (30%): 正样本应在排名靠前
         - 在 Top-3 → 重奖 (越靠前奖越重)
         - 在中段   → 轻奖
         - 在后段   → 惩罚
         - 未出现   → 惩罚 (比放错位置更差)
      3. context (10%): 非大促期 MSA 卡片不应出现

    位置权重: 1/(i+1) — 越靠前信号越强
    """
    n = len(rank_list)
    if n == 0:
        return {"score": 0.0, "neg_avg_position": 0.0, "pos_avg_position": 0.0}

    # ── 1. Negative score: 负样本应在排名靠后 ──
    neg_score = 0.0
    neg_positions = []
    if negative_keys:
        for key in negative_keys:
            if key in rank_list:
                pos = rank_list.index(key)
                neg_positions.append(pos)
                weight = 1.0 / (pos + 1)  # 位置权重: pos0=1.0, pos1=0.5, pos2=0.33, pos15=0.0625
                if pos < 3:
                    neg_score += -weight      # Top-3: 重罚
                elif pos < n // 2:
                    neg_score += -weight * 0.3  # 中段: 轻罚
                else:
                    neg_score += weight * 0.5   # 后段: 奖励
            # else: 未出现 → 0 (中性)
        neg_score /= len(negative_keys)
    else:
        neg_score = 0.0
    neg_avg_pos = sum(neg_positions) / len(neg_positions) if neg_positions else -1

    # ── 2. Positive score: 正样本应在排名靠前 ──
    pos_score = 0.0
    pos_positions = []
    if positive_keys:
        for key in positive_keys:
            if key in rank_list:
                pos = rank_list.index(key)
                pos_positions.append(pos)
                weight = 1.0 / (pos + 1)
                if pos < 3:
                    pos_score += weight          # Top-3: 重奖
                elif pos < n // 2:
                    pos_score += weight * 0.5    # 中段: 轻奖
                else:
                    pos_score += -weight * 0.3   # 后段: 惩罚
            else:
                pos_score += -0.3                # 未出现: 惩罚
        pos_score /= len(positive_keys)
    else:
        pos_score = 0.0
    pos_avg_pos = sum(pos_positions) / len(pos_positions) if pos_positions else -1

    # ── 3. Context score: 非大促期 MSA 卡片不应出现 ──
    ctx_score = 0.0
    is_mega = ctx.get("is_mega_period", False)
    if not is_mega:
        msa_keywords = ["msa", "mega", "sd_msa"]
        msa_count = 0
        for key in rank_list:
            for kw in msa_keywords:
                if kw in key.lower():
                    msa_count += 1
                    break
        if msa_count > 0:
            ctx_score = -0.5  # 有 MSA 卡片出现 → 惩罚
        else:
            ctx_score = 0.5   # 无 MSA 卡片 → 奖励

    # ── 加权求和 ──
    if positive_keys:
        score = 0.6 * neg_score + 0.3 * pos_score + 0.1 * ctx_score
    else:
        score = 0.7 * neg_score + 0.3 * ctx_score

    score = max(-1.0, min(1.0, score))

    return {
        "score": round(score, 4),
        "neg_avg_position": round(neg_avg_pos, 2),
        "pos_avg_position": round(pos_avg_pos, 2),
        "neg_score": round(neg_score, 4),
        "pos_score": round(pos_score, 4),
        "ctx_score": round(ctx_score, 4),
    }


# ============================================================================
# Main Entry: compute_reward
# ============================================================================

def compute_reward(
    solution_str: str,
    ground_truth: Dict[str, Any],
    extra_info: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Knowledge-Grounded reward computation (v5 — unified formula).

    统一公式:
      final_score = α × rule_score + (1-α) × judge_score + format_penalty

      - 有 LLM Judge: α=0.5 → final = 0.5×rule + 0.5×judge + format_penalty
      - 无 LLM Judge: α=1.0 → final = rule_score + format_penalty

    格式惩罚:
      - 完全无效（0张卡解析出）: -0.3
      - 部分有效（<3张卡）: -0.15
      - 合法输出的格式问题: markdown -0.1, 幻觉 -0.15/个, 遗漏 -0.025/个

    Returns:
        {"score": float, "format_valid": bool, "format_penalty": float,
         "rule_score": float, "judge_score": float, ...}
    """
    card_pool = ground_truth.get("card_pool", [])
    positive_keys = ground_truth.get("positive_keys", []) or ground_truth.get("positive_actions", [])
    negative_keys = ground_truth.get("negative_keys", [])
    ctx = ground_truth.get("context", {})
    ctx["card_pool"] = card_pool
    ctx["positive_actions"] = positive_keys

    # ── 1. Format Check ──
    fc = format_check(solution_str, card_pool)
    rank_list = fc["rank_list"]

    # ── 2. Rule-based Score（所有情况都算，用已解析的部分） ──
    rule_result = _rule_based_score(rank_list, positive_keys, negative_keys, ctx)
    rule_score = rule_result["score"]

    # ── 3. Format Penalty ──
    if not fc["valid"]:
        if len(rank_list) == 0:
            format_penalty = -0.3   # 完全无效
        else:
            format_penalty = -0.15  # 部分有效但不足 3 张
    else:
        format_penalty = fc["penalty"]  # 格式合法，可能有 markdown/幻觉/遗漏惩罚

    # ── 4. LLM Judge（格式合法时才有意义） ──
    judge_score = 0.0
    judge_result = {}
    has_judge = False

    if fc["valid"]:
        client = _get_llm_client()
        if client:
            judge_model = os.environ.get("JUDGE_MODEL", "qwen-plus")
            query_model = os.environ.get("QUERY_MODEL", "qwen-turbo")

            # TipBank Retrieval
            tips = _retrieve_tips(ctx, card_pool, client, query_model)

            # Intent Recognition
            from .intent_recognizer import recognize_intent
            intent = recognize_intent(ctx, client, judge_model)

            # Knowledge Judge
            from .knowledge_judge import knowledge_judge_single
            judge_result = knowledge_judge_single(
                rank_list, ctx, tips, intent,
                positive_keys, negative_keys,
                client, judge_model,
            )
            judge_score = judge_result.get("score", 0.0)
            has_judge = True

    # ── 5. 统一公式 ──
    if has_judge:
        # 有 judge: 50% rule + 50% judge + format_penalty
        final_score = 0.5 * rule_score + 0.5 * judge_score + format_penalty
    else:
        # 无 judge: rule + format_penalty
        final_score = rule_score + format_penalty

    return {
        "score": round(final_score, 4),
        "format_valid": fc["valid"],
        "format_penalty": round(format_penalty, 4),
        "has_markdown": fc.get("has_markdown", False),
        "n_hallucinated": fc.get("n_hallucinated", 0),
        "n_missing": fc.get("n_missing", 0),
        "rule_score": round(rule_score, 4),
        "rule_neg_position": rule_result.get("neg_avg_position", 0.0),
        "rule_pos_position": rule_result.get("pos_avg_position", 0.0),
        "judge_score": round(judge_score, 4),
        "intent_alignment": judge_result.get("intent_alignment", 0.0),
        "strategy_compliance": judge_result.get("strategy_compliance", 0.0),
        "result_prediction": judge_result.get("result_prediction", 0.0),
        "risk_avoidance": judge_result.get("risk_avoidance", 0.0),
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
