"""LLM Judge: batch 并发调用 LLM 评判排序合理性.

使用 .env 中的 OPENAI_API_KEY / OPENAI_BASE_URL 配置。
支持 async batch 并发，适合 veRL reward 计算。

用法:
  # 单条测试
  python -m grpo.llm_judge --test

  # batch 模式 (veRL 集成时由 reward worker 调用)
  scores = asyncio.run(batch_llm_judge(rank_lists, contexts, llm_config))
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

LLM_JUDGE_SYSTEM = """你是 Lazada 广告运营诊断专家。请评判给定的卡片排序是否合理地帮助了商家。

核心原则：推荐应基于商家当前真实需求，不是一味让商家花更多钱。

## 评判规则（按优先级）

### 1. 资金风险对齐 (最高优先)
- OOBA 风险 (balance/l7d_avg < 3) → 充值类应在 Top-3
- 无 OOBA 风险 → 充值类不应在 Top-3
- OOBU 风险 (budget_util > 85%) → 对应类型提预算应在 Top-5

### 2. 商家意图一致性
- 商家近期在扩量 (提预算/降TROI) → 扩量类可排前
- 商家近期在控成本 (降预算/提TROI) → 不该推加预算/极速起量
- 行为方向与排序冲突 → 严重扣分

### 3. 计划存在性
- 计划不存在 → 操作类卡(非 creation)不应在 Top-5
- 计划存在且活跃 → 创建类不应在 Top-3

### 4. 预算/出价水位排序
- 预算使用率 > 90% → budget 应排在 troi 前面
- 预算使用率 < 70% → troi/max_cpc 应排在 budget 前面

### 5. 场景约束
- 非大促期 → MSA 相关不应在 Top-5
- 广告表现差 → 出价优化类应排前

## 评分标准
- 0.8~1.0: 完全合理
- 0.4~0.7: 基本合理，有小瑕疵
- 0.0~0.3: 有明显问题
- -0.5~-0.1: 严重误推
- -1.0~-0.5: 完全违背商家需求

输出一个 JSON: {"score": 0.7, "reason": "一句话总结"}
score 范围 [-1.0, 1.0]。不要输出其他内容。"""


LLM_JUDGE_USER = """## 商家近期行为
{seller_behavior}

## 商家钱包数据
{wallet_data}

## 商家广告投放效果
{campaign_data}

## 候选卡片池
{card_pool}

## 模型排序结果 (Top-8)
{rank_list}

## 已知结果 (历史验证)
- 商家随后做了正向操作: {positive_actions}
- 商家随后做了反向操作: {negative_actions}

请按规则评判此排序的质量。"""


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

def get_llm_config() -> Dict[str, str]:
    """从环境变量读取 LLM 配置."""
    return {
        "api_key": os.environ.get("OPENAI_API_KEY", ""),
        "base_url": os.environ.get("OPENAI_BASE_URL", ""),
        "model": os.environ.get("OPENAI_MODEL", "qwen-plus"),
    }


# ---------------------------------------------------------------------------
# 单条调用
# ---------------------------------------------------------------------------

def build_judge_messages(
    rank_list: List[str],
    context: Dict[str, Any],
    positive_keys: Optional[List[str]] = None,
    negative_keys: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    """构建 LLM Judge 的 messages."""
    user_msg = LLM_JUDGE_USER.format(
        seller_behavior=(context.get("seller_behavior") or "无")[:500],
        wallet_data=(context.get("wallet_data") or "无")[:400],
        campaign_data=(context.get("campaign_data") or "无")[:600],
        card_pool=json.dumps(context.get("card_pool", rank_list)[:12], ensure_ascii=False),
        rank_list=json.dumps(rank_list[:8], ensure_ascii=False),
        positive_actions=json.dumps(positive_keys, ensure_ascii=False) if positive_keys else "无",
        negative_actions=json.dumps(negative_keys, ensure_ascii=False) if negative_keys else "无",
    )
    return [
        {"role": "system", "content": LLM_JUDGE_SYSTEM},
        {"role": "user", "content": user_msg},
    ]


def parse_judge_response(raw: str) -> float:
    """解析 LLM Judge 的 JSON 响应."""
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s).strip()

    # 尝试从 thinking 标签后提取
    if "</think>" in s:
        s = s.split("</think>")[-1].strip()

    try:
        parsed = json.loads(s)
        score = float(parsed.get("score", 0))
        return max(-1.0, min(1.0, score))
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # fallback: 尝试正则提取 score
    m = re.search(r'"score"\s*:\s*([-\d.]+)', s)
    if m:
        try:
            return max(-1.0, min(1.0, float(m.group(1))))
        except ValueError:
            pass

    logger.warning("LLM Judge 响应解析失败: %s", s[:100])
    return 0.0


# ---------------------------------------------------------------------------
# Batch 异步调用
# ---------------------------------------------------------------------------

async def batch_llm_judge(
    rank_lists: List[List[str]],
    contexts: List[Dict[str, Any]],
    positive_keys_list: Optional[List[List[str]]] = None,
    negative_keys_list: Optional[List[List[str]]] = None,
    llm_config: Optional[Dict[str, str]] = None,
    max_concurrent: int = 30,
) -> List[float]:
    """批量并发调用 LLM Judge.

    Args:
        rank_lists: 每个 rollout 的排序结果
        contexts: 每个 rollout 对应的上下文
        positive_keys_list: 每条的 positive keys (可选)
        negative_keys_list: 每条的 negative keys (可选)
        llm_config: LLM 配置, 默认从环境变量读取
        max_concurrent: 最大并发数

    Returns:
        list of float scores, 与输入等长
    """
    from openai import AsyncOpenAI

    cfg = llm_config or get_llm_config()
    if not cfg.get("api_key"):
        logger.error("LLM Judge 需要 OPENAI_API_KEY")
        return [0.0] * len(rank_lists)

    client = AsyncOpenAI(
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        timeout=60.0,
        max_retries=2,
    )
    model = cfg.get("model", "qwen-plus")
    semaphore = asyncio.Semaphore(max_concurrent)

    pos_list = positive_keys_list or [None] * len(rank_lists)
    neg_list = negative_keys_list or [None] * len(rank_lists)

    async def _judge_one(idx: int) -> float:
        async with semaphore:
            try:
                messages = build_judge_messages(
                    rank_lists[idx],
                    contexts[idx],
                    pos_list[idx],
                    neg_list[idx],
                )
                resp = await client.chat.completions.create(
                    model=model,
                    temperature=0.1,
                    max_tokens=200,
                    messages=messages,
                    extra_body={
                        "enable_thinking": False,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                )
                raw = resp.choices[0].message.content or ""
                return parse_judge_response(raw)
            except Exception as e:
                logger.warning("LLM Judge 调用失败 [%d]: %s", idx, e)
                return 0.0

    tasks = [_judge_one(i) for i in range(len(rank_lists))]
    results = await asyncio.gather(*tasks)
    return list(results)


# ---------------------------------------------------------------------------
# 同步包装
# ---------------------------------------------------------------------------

def judge_single(
    rank_list: List[str],
    context: Dict[str, Any],
    positive_keys: Optional[List[str]] = None,
    negative_keys: Optional[List[str]] = None,
    llm_config: Optional[Dict[str, str]] = None,
) -> float:
    """同步调用单条 LLM Judge."""
    from openai import OpenAI

    cfg = llm_config or get_llm_config()
    if not cfg.get("api_key"):
        return 0.0

    client = OpenAI(
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        timeout=60.0,
        max_retries=2,
    )
    messages = build_judge_messages(rank_list, context, positive_keys, negative_keys)

    try:
        resp = client.chat.completions.create(
            model=cfg.get("model", "qwen-plus"),
            temperature=0.1,
            max_tokens=200,
            messages=messages,
            extra_body={
                "enable_thinking": False,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        raw = resp.choices[0].message.content or ""
        return parse_judge_response(raw)
    except Exception as e:
        logger.warning("LLM Judge 单条调用失败: %s", e)
        return 0.0


# ---------------------------------------------------------------------------
# 测试入口
# ---------------------------------------------------------------------------

def _test():
    """快速测试 LLM Judge."""
    logging.basicConfig(level=logging.INFO)

    # 模拟数据
    rank_list = [
        "smax_am_x_budget_increase", "smax_am_x_troi", "manual_topup",
        "smax_am_creation", "smax_aa_creation", "sd_am_budget_increase",
    ]
    context = {
        "wallet_data": '{"balance":"500000","currency":"idr","l7d_spend":"200000"}',
        "campaign_data": '{"summary":{"smax_am_x_count":2,"sd_am_count":3},'
                         '"smax_am_x_campaigns":[{"id":1,"perf_today":{"budget_util":"95%"}}]}',
        "seller_behavior": "[2026-05-17] Change Campaign Setting: budget from 25000 to 50000",
        "card_pool": rank_list + ["auto_topup", "smax_am_x_rapid_boost"],
    }

    score = judge_single(
        rank_list, context,
        positive_keys=["smax_am_x_budget_increase"],
        negative_keys=["smax_am_x_troi"],
    )
    print(f"LLM Judge score: {score}")


if __name__ == "__main__":
    if "--test" in sys.argv:
        _test()
    else:
        print("Usage: python -m grpo.llm_judge --test")
