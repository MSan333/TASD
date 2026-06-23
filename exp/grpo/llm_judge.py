"""Batch Knowledge-Grounded Judge for GRPO reward.

优化策略:
  - 同一 sample 的 K 个 rollout 共享 TipBank 检索和意图识别
    (context 相同, 只有 rank_list 不同)
    → K 个 rollout 只需 1 次检索 + 1 次意图识别 + K 次打分
  - async 并发 LLM 调用, semaphore 控制并发数
  - 检索缓存: 相同 context 只检索一次
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# Helpers
# ============================================================================

def _context_key(gt: Dict) -> str:
    """Generate a cache key for context (same seller+time → same retrieval)."""
    ctx = gt.get("context", {})
    raw = f"{ctx.get('venture', '')}|{ctx.get('seller_id', '')}|{ctx.get('seller_behavior', '')[:200]}"
    return hashlib.md5(raw.encode()).hexdigest()


# ============================================================================
# Batch Knowledge Judge (async)
# ============================================================================

async def batch_knowledge_judge(
    rank_lists: List[List[str]],
    ground_truths: List[Dict],
    max_concurrent: int = 20,
) -> List[Dict]:
    """Batch async Knowledge-Grounded Judge.

    Optimization: group by context_key so same-sample rollouts share
    TipBank retrieval and intent recognition (only judge scoring differs).

    Returns List[Dict] — each dict has "score" + sub-metrics
    (intent_alignment, strategy_compliance, result_prediction, risk_avoidance).
    """
    from openai import AsyncOpenAI, OpenAI

    from .intent_recognizer import recognize_intent
    from .knowledge_judge import (
        format_tips_for_judge,
        parse_judge_response,
        JUDGE_PROMPT,
        _get_matching_prompt,
    )

    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "")
    judge_model = os.environ.get("JUDGE_MODEL", "qwen-plus")
    query_model = os.environ.get("QUERY_MODEL", "qwen-turbo")

    if not api_key:
        return [{"score": 0.0} for _ in rank_lists]

    async_client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=60.0, max_retries=2)
    sync_client = OpenAI(api_key=api_key, base_url=base_url)
    semaphore = asyncio.Semaphore(max_concurrent)

    # --- Phase 1: Group by context, compute shared retrieval + intent ---
    context_cache: Dict[str, Dict] = {}

    tip_index = None
    try:
        tipbank_path = os.environ.get("TIPBANK_PATH", "")
        if tipbank_path and os.path.exists(tipbank_path):
            from .tipbank_loader import load_tipbank, TipIndex, generate_search_queries
            tips = load_tipbank(tipbank_path)
            if tips:
                tip_index = TipIndex(tips)
    except Exception as e:
        logger.warning("TipBank setup failed for batch: %s", e)

    for gt in ground_truths:
        key = _context_key(gt)
        if key in context_cache:
            continue

        ctx = gt.get("context", {})
        card_pool = gt.get("card_pool", [])
        positive_keys = gt.get("positive_keys", [])
        ctx["card_pool"] = card_pool
        ctx["positive_actions"] = positive_keys

        # Step 1: Retrieve tips (once per unique context)
        tips_result = []
        if tip_index and sync_client:
            try:
                from .tipbank_loader import generate_search_queries
                queries = generate_search_queries(
                    sync_client, query_model, ctx, card_pool,
                    extra_context=f"商家随后执行了: {positive_keys}",
                )
                tips_result = tip_index.search(
                    queries=queries, top_k=15,
                    venture=ctx.get("venture", ""),
                    action_names=card_pool,
                )
            except Exception as e:
                logger.debug("TipBank retrieval failed for key %s: %s", key[:8], e)

        # Step 2: Recognize intent (once per unique context)
        intent = recognize_intent(ctx, sync_client, judge_model)

        context_cache[key] = {
            "tips": tips_result,
            "intent": intent,
            "ctx": ctx,
        }

    # Load matching skill (once)
    matching_skill = _get_matching_prompt() or "（无匹配准则）"

    # --- Phase 2: Async judge scoring for each rank_list ---
    async def _judge_one(idx: int) -> Dict:
        """返回完整 judge result dict (score + 各维度子分数)."""
        async with semaphore:
            gt = ground_truths[idx]
            key = _context_key(gt)
            cached = context_cache.get(key, {})
            tips_result = cached.get("tips", [])
            intent = cached.get("intent", {})
            ctx = cached.get("ctx", gt.get("context", {}))

            pos = gt.get("positive_keys", [])
            neg = gt.get("negative_keys", [])
            card_pool = gt.get("card_pool", rank_lists[idx])

            # 手动替换避免 skill/tips 内容中的 {} 干扰 .format()
            prompt = JUDGE_PROMPT.replace("{retrieved_tips}", format_tips_for_judge(tips_result))
            prompt = prompt.replace("{intent_result}", json.dumps(intent, ensure_ascii=False))
            prompt = prompt.replace("{matching_skill}", matching_skill)
            prompt = prompt.replace("{rank_list}", json.dumps(rank_lists[idx], ensure_ascii=False))
            prompt = prompt.replace("{positive_actions}", json.dumps(pos, ensure_ascii=False))
            prompt = prompt.replace("{negative_actions}", json.dumps(neg, ensure_ascii=False))
            prompt = prompt.replace("{card_pool}", json.dumps(card_pool, ensure_ascii=False))

            try:
                resp = await async_client.chat.completions.create(
                    model=judge_model,
                    temperature=0.2,
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}],
                    extra_body={"enable_thinking": False},
                )
                raw = resp.choices[0].message.content or ""
                result = parse_judge_response(raw)
                result.setdefault("score", 0.0)
                return result
            except Exception as e:
                logger.warning("Judge call failed [%d]: %s", idx, e)
                return {"score": 0.0}

    tasks = [_judge_one(i) for i in range(len(rank_lists))]
    results = await asyncio.gather(*tasks)

    scores_only = [r.get("score", 0.0) for r in results]
    valid_scores = [s for s in scores_only if s != 0.0]
    avg = sum(valid_scores) / max(len(valid_scores), 1) if valid_scores else 0.0
    logger.info(
        "Batch Knowledge Judge: total=%d, contexts=%d, avg_score=%.4f",
        len(results), len(context_cache), avg,
    )

    return list(results)
