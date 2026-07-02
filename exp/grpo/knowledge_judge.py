"""Knowledge-Grounded Judge — 知识驱动排序质量评分 (v10).

三步串行:
  Step 1: TipBank 检索 (复用 tipbank_loader)
  Step 2: 商家意图识别 (复用 intent_recognizer)
  Step 3: 知识对比打分 — LLM 结合 tips + intent + matching skill + rank_list 综合评判

评判维度 (v10, 语义主导):
  A. 标签吻合 (20%) — 结果信号, 低权重避免与 rule_score 重叠
  B. 行为语义一致性 (40%) — 排序与商家行为方向是否匹配
  C. 风险规避 (40%) — 避免危险推荐, 与商家状态匹配

v10 校准改进:
  v9 问题: avg_judge=-0.02, Judge 无区分度
  v10: 明确指导打分范围, 合理排序应给正分
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ============================================================================
# Matching Skill 加载 (进程级单例)
# ============================================================================

_MATCHING_PROMPT: Optional[str] = None


def _get_matching_prompt() -> str:
    global _MATCHING_PROMPT
    if _MATCHING_PROMPT is not None:
        return _MATCHING_PROMPT

    skills_dir = os.path.join(os.path.dirname(__file__), "skills")
    if not os.path.isdir(skills_dir):
        _MATCHING_PROMPT = ""
        return _MATCHING_PROMPT

    try:
        from .skills import SkillLoader
        loader = SkillLoader(skills_dir)
        _MATCHING_PROMPT = loader.get_matching_prompt()
        if _MATCHING_PROMPT:
            logger.info("Matching skill loaded for Knowledge Judge")
    except Exception as e:
        logger.warning("Failed to load matching skill: %s", e)
        _MATCHING_PROMPT = ""

    return _MATCHING_PROMPT

JUDGE_PROMPT = """\
你是排序质量评判专家。请基于以下信息，全面评判模型的排序结果。

## 1. 模型的排序输出
{rank_list}

## 2. 商家实际后续行为（ground truth）
- 正向操作（商家确实执行了）: {positive_actions}
- 负向操作（商家做了相反的操作）: {negative_actions}

## 3. 商家上下文
{context_summary}

## 4. 候选卡片池
{card_pool}

{reference_knowledge}

## 评判维度

### A. 标签吻合 (20%)
模型排序与商家实际行为的吻合度（只关注 Top-3，因为线上只曝光前 3 位）：
- 正向操作排在 Top-3 → 加分
- 负向操作排在 Top-3 → 扣分（推荐了商家明确不需要的操作）
- 如果标签为空，此维度给中性分

注意: 此维度权重低, 因为标签只是结果信号, 不应主导评判。

### B. 行为语义一致性 (40%)
从商家近期行为日志推断其意图方向，判断排序是否合理：
- 商家提高预算/新建计划/降低troi → 扩量意图 → 排序应推扩量类操作卡
- 商家降低预算/关闭计划/提高troi → 控制成本意图 → 排序应推优化类/省钱卡
- Top-3 整体方向与行为方向一致 → 加分
- Top-3 整体方向与行为方向矛盾 → 扣分
- 无法明确判断 → 中性分

关键: 即使标签吻合度高, 如果排序与商家行为语义矛盾, 应给低分。
关注排序的**整体逻辑**而非单个位置。

### C. 风险与合理性 (40%)
综合判断排序是否存在风险或不合理：
- 资金风险(OOBA)下推了花钱卡而非充值卡 → 严重扣分
- 无活跃计划时推了操作卡（如调 bid/budget）→ 扣分
- 大促期推了 MSA 卡 → 合理; 非大促期推 MSA 卡 → 不合理
- 排序整体逻辑与商家当前状态匹配 → 加分
- 如果发现规则系统的判断与商家实际情况矛盾，按独立判断给分

关键: 这是**安全性维度**, 推荐了危险操作应给很低分, 推荐了安全合理操作应给高分。

## 打分指导 (最重要, 必须遵守)

**核心要求: 你的打分必须区分不同排序的质量差异。绝对不要所有输出都给类似分数。**

**打分校准 (v10, 严格遵守):**
- **+0.5 到 +1.0**: 排序语义合理、无风险、标签吻合度高 (优秀排序)
- **+0.2 到 +0.5**: 排序方向基本正确, 有小瑕疵但不严重 (良好排序)
- **0.0 到 +0.2**: 排序中性, 无明显优劣 (平庸排序)
- **-0.3 到 0.0**: 排序有小问题, 方向略偏 (较差排序)
- **-1.0 到 -0.3**: 排序存在严重语义矛盾或明显风险 (失败排序)

**重要校准原则:**
1. **大部分合理排序应给正分**: 只要排序方向正确、无明显风险, 应给 +0.2 以上
2. **负分仅用于严重问题**: 只有推荐了危险操作 (如 OOBA 推花钱卡) 或完全与商家状态矛盾时才给负分
3. **避免集中在 0 附近**: 如果你的打分集中在 [-0.1, +0.1], 说明你没有充分区分质量差异
4. **参考分布**: 一个良好训练的模型, 大部分排序应落在 [+0.2, +0.6], 少数优秀 >+0.7, 少数失败 <-0.3

注意:
- 即使排序不完美, 只要方向正确、无明显风险, 应给正分
- 只有存在**严重问题**(推荐危险操作、与商家状态完全矛盾)时才给负分
- 最终 score 是三个维度的加权综合分

请直接输出 JSON（不要输出其他内容）:
{{
  "score": -1.0 到 1.0,
  "result_prediction": 0.0-1.0,
  "behavior_alignment": 0.0-1.0,
  "risk_assessment": 0.0-1.0,
  "rule_disagreement": "如果你认为规则系统的判断有误, 简要说明; 否则留空",
  "reasoning": "一句话总结"
}}
"""


def format_tips_for_judge(tips: list) -> str:
    """将检索到的 tips 格式化为 Judge 可读文本."""
    if not tips:
        return "（无匹配策略）"
    lines = []
    for i, tip in enumerate(tips[:10], 1):
        name = getattr(tip, 'name', '') if hasattr(tip, 'name') else tip.get('name', '')
        principle = getattr(tip, 'principle', '') if hasattr(tip, 'principle') else tip.get('principle', '')
        when = getattr(tip, 'when_to_apply', '') if hasattr(tip, 'when_to_apply') else tip.get('when_to_apply', '')
        how = getattr(tip, 'how_to_apply', '') if hasattr(tip, 'how_to_apply') else tip.get('how_to_apply', '')
        conf = getattr(tip, 'confidence', 0) if hasattr(tip, 'confidence') else tip.get('confidence', 0)
        lines.append(
            f"{i}. **{name}** [置信度:{conf:.2f}]\n"
            f"   原则: {principle}\n"
            f"   适用条件: {when}\n"
            f"   操作方法: {how}"
        )
    return "\n".join(lines)


def parse_judge_response(raw: str) -> Dict[str, Any]:
    """解析 Judge LLM 响应, 提取评分 JSON."""
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s).strip()
    if "</think>" in s:
        s = s.split("</think>")[-1].strip()

    try:
        match = re.search(r'\{.*\}', s, re.DOTALL)
        if match:
            result = json.loads(match.group())
            result["score"] = max(-1.0, min(1.0, float(result.get("score", 0))))
            for key in ("result_prediction", "behavior_alignment", "risk_assessment"):
                if key in result:
                    result[key] = max(0.0, min(1.0, float(result[key])))
            return result
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    m = re.search(r'"score"\s*:\s*([-\d.]+)', s)
    if m:
        try:
            return {"score": max(-1.0, min(1.0, float(m.group(1)))), "reasoning": "partial_parse"}
        except ValueError:
            pass

    return {"score": 0.0, "reasoning": "parse_failed"}


def _build_context_summary(ctx: Dict[str, Any]) -> str:
    """从 ctx 中提取简洁的商家上下文摘要, 供 Judge 理解语义."""
    parts = []

    # 商家近期行为 (截取前 500 字符, 避免 prompt 过长)
    behavior = ctx.get("seller_behavior", "")
    if isinstance(behavior, str) and behavior.strip():
        # 只保留最近的行为记录
        lines = behavior.strip().split("\\n")
        recent = "\\n".join(lines[-5:]) if len(lines) > 5 else behavior.strip()
        parts.append(f"近期行为:\n{recent[:500]}")

    # 钱包/余额状态
    wallet = ctx.get("wallet_data", "")
    if isinstance(wallet, str) and wallet.strip():
        parts.append(f"钱包状态: {wallet[:200]}")
    elif isinstance(wallet, dict):
        parts.append(f"钱包状态: {json.dumps(wallet, ensure_ascii=False)[:200]}")

    # rule_signals 中的关键状态
    rs = ctx.get("rule_signals", {})
    if rs:
        signals = []
        if rs.get("has_ooba"):
            signals.append("⚠️ 余额不足风险(OOBA)")
        if rs.get("is_mega_period"):
            signals.append("🎉 大促期")
        oobu = rs.get("oobu_plans", [])
        if hasattr(oobu, 'size'):
            oobu_list = list(oobu) if oobu.size > 0 else []
        else:
            oobu_list = list(oobu) if oobu else []
        if oobu_list:
            signals.append(f"预算撞线计划: {', '.join(str(x) for x in oobu_list)}")
        existing = rs.get("existing_plans", [])
        if hasattr(existing, 'size'):
            existing_list = list(existing) if existing.size > 0 else []
        else:
            existing_list = list(existing) if existing else []
        if existing_list:
            signals.append(f"已有计划: {', '.join(str(x) for x in existing_list)}")
        if signals:
            parts.append("关键状态: " + "; ".join(signals))

    return "\n\n".join(parts) if parts else "（无上下文信息）"


def _build_reference_knowledge(
    tips: Optional[list] = None,
    intent: Optional[Dict[str, Any]] = None,
) -> str:
    """构建参考知识段, 注入 Judge prompt 作为背景信息.

    Tips 和意图不作为评分维度, 而是让 Judge 在评判"语义合理性"时有更多参考.
    如果检索/识别质量高, Judge 会利用; 如果质量低, Judge 会忽略.
    """
    parts = []

    if tips:
        tip_lines = []
        for i, tip in enumerate(tips[:8], 1):
            name = getattr(tip, 'name', '') if hasattr(tip, 'name') else tip.get('name', '')
            principle = getattr(tip, 'principle', '') if hasattr(tip, 'principle') else tip.get('principle', '')
            when = getattr(tip, 'when_to_apply', '') if hasattr(tip, 'when_to_apply') else tip.get('when_to_apply', '')
            how = getattr(tip, 'how_to_apply', '') if hasattr(tip, 'how_to_apply') else tip.get('how_to_apply', '')
            tip_lines.append(f"  {i}. **{name}**: {principle} (适用: {when}; 方法: {how})")
        if tip_lines:
            parts.append("### 参考: 相关专家策略\n" + "\n".join(tip_lines))

    if intent:
        primary = intent.get("primary_intent", "")
        confidence = intent.get("confidence", 0)
        evidence = intent.get("evidence", "")
        direction = intent.get("recommended_direction", "")
        if primary:
            intent_text = f"### 参考: 商家意图分析\n  主要意图: {primary} (置信度: {confidence:.2f})"
            if evidence:
                intent_text += f"\n  依据: {str(evidence)[:200]}"
            if direction:
                intent_text += f"\n  建议方向: {direction}"
            parts.append(intent_text)

    if not parts:
        return ""
    return "## 5. 参考知识（辅助判断，不作为评分维度）\n" + "\n\n".join(parts)


def build_judge_prompt(
    rank_list: List[str],
    ctx: Dict[str, Any],
    positive_keys: List[str],
    negative_keys: List[str],
    tips: Optional[list] = None,
    intent: Optional[Dict[str, Any]] = None,
) -> str:
    """构建 Judge prompt.

    三个评判维度:
      - 结果预测 (70%): 商家实际行为 — 硬信号
      - 语义合理性 (20%): 排序与商家上下文是否一致 (可参考 TipBank/意图)
      - 风险规避 (10%): 避免危险推荐

    tips 和 intent 作为参考知识注入, 不作为独立评分维度.
    """
    card_pool = ctx.get("card_pool", rank_list)
    context_summary = _build_context_summary(ctx)
    reference_knowledge = _build_reference_knowledge(tips, intent)

    prompt = JUDGE_PROMPT
    prompt = prompt.replace("{rank_list}", json.dumps(rank_list, ensure_ascii=False))
    prompt = prompt.replace("{positive_actions}", json.dumps(positive_keys, ensure_ascii=False))
    prompt = prompt.replace("{negative_actions}", json.dumps(negative_keys, ensure_ascii=False))
    prompt = prompt.replace("{card_pool}", json.dumps(card_pool, ensure_ascii=False))
    prompt = prompt.replace("{context_summary}", context_summary)
    prompt = prompt.replace("{reference_knowledge}", reference_knowledge)
    return prompt


def knowledge_judge_single(
    rank_list: List[str],
    ctx: Dict[str, Any],
    positive_keys: List[str],
    negative_keys: List[str],
    client,
    model: str,
    tips: Optional[list] = None,
    intent: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """对单个 rank_list 执行打分."""
    prompt = build_judge_prompt(
        rank_list, ctx, positive_keys, negative_keys, tips, intent,
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0.2,
            max_tokens=500,
            timeout=30,
            messages=[{"role": "user", "content": prompt}],
            extra_body={"enable_thinking": False},
        )
        content = (resp.choices[0].message.content or "").strip()
        return parse_judge_response(content)
    except Exception as e:
        logger.warning("Knowledge Judge failed: %s", e)
        return {"score": 0.0, "reasoning": f"judge_error: {e}"}


# ============================================================================
# Pairwise Judge (v11) — 相对比较替代绝对打分
# ============================================================================

PAIRWISE_JUDGE_PROMPT = """\
你是排序质量评判专家。请比较两个排序方案，判断哪个更好。

## 商家上下文
{context_summary}

## 商家实际行为
- 正向操作（商家确实执行了）: {positive_actions}
- 负向操作（商家做了相反的操作）: {negative_actions}

## 候选卡片池
{card_pool}

## 排序 A
{rank_a}

## 排序 B
{rank_b}

## 评判标准
请判断排序 A 相对于排序 B 的质量：
1. **与商家意图一致性**：哪个排序更符合商家实际行为方向？
2. **推荐合理性**：哪个排序的 Top-3 更合理、更安全？
3. **避免错误推荐**：哪个排序更少推荐商家不需要的操作？

## 输出格式
只输出 JSON，不要其他内容：
{{
  "preference": 1, 0, 或 -1
}}

- preference = 1: 排序 A 明显更好
- preference = 0: 两者质量相近
- preference = -1: 排序 B 明显更好
"""


def parse_pairwise_response(raw: str) -> int:
    """解析 pairwise judge 响应，返回 preference ∈ {-1, 0, 1}."""
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s).strip()
    if "</think>" in s:
        s = s.split("</think>")[-1].strip()

    try:
        match = re.search(r'\{.*\}', s, re.DOTALL)
        if match:
            result = json.loads(match.group())
            pref = int(result.get("preference", 0))
            if pref in (-1, 0, 1):
                return pref
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Fallback: 尝试直接解析数字
    m = re.search(r'(-?\d+)', s)
    if m:
        pref = int(m.group(1))
        if pref in (-1, 0, 1):
            return pref

    return 0


def pairwise_compare(
    rank_a: List[str],
    rank_b: List[str],
    ctx: Dict[str, Any],
    positive_keys: List[str],
    negative_keys: List[str],
    client,
    model: str,
) -> int:
    """同步 pairwise 比较: 返回 +1 (A>B), 0 (A≈B), -1 (B>A)."""
    context_summary = _build_context_summary(ctx)

    prompt = PAIRWISE_JUDGE_PROMPT
    prompt = prompt.replace("{rank_a}", json.dumps(rank_a, ensure_ascii=False))
    prompt = prompt.replace("{rank_b}", json.dumps(rank_b, ensure_ascii=False))
    prompt = prompt.replace("{positive_actions}", json.dumps(positive_keys, ensure_ascii=False))
    prompt = prompt.replace("{negative_actions}", json.dumps(negative_keys, ensure_ascii=False))
    prompt = prompt.replace("{card_pool}", json.dumps(ctx.get("card_pool", []), ensure_ascii=False))
    prompt = prompt.replace("{context_summary}", context_summary)

    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0.1,
            max_tokens=100,
            timeout=30,
            messages=[{"role": "user", "content": prompt}],
            extra_body={"enable_thinking": False},
        )
        content = (resp.choices[0].message.content or "").strip()
        return parse_pairwise_response(content)
    except Exception as e:
        logger.warning("Pairwise judge failed: %s", e)
        return 0


async def async_pairwise_compare(
    pairs: List[tuple],
    max_concurrent: int = 10,
) -> List[int]:
    """异步批量 pairwise 比较.

    Args:
        pairs: [(rank_a, rank_b, ctx, pos_keys, neg_keys), ...]
        max_concurrent: 最大并发数

    Returns:
        List[int]: 每对的 preference ∈ {-1, 0, 1}
    """
    from openai import AsyncOpenAI

    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "")
    model = os.environ.get("JUDGE_MODEL", "qwen-plus")

    if not api_key:
        return [0] * len(pairs)

    client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=60.0, max_retries=2)
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _judge_one(pair: tuple) -> int:
        async with semaphore:
            rank_a, rank_b, ctx, pos_keys, neg_keys = pair
            context_summary = _build_context_summary(ctx)

            prompt = PAIRWISE_JUDGE_PROMPT
            prompt = prompt.replace("{rank_a}", json.dumps(rank_a, ensure_ascii=False))
            prompt = prompt.replace("{rank_b}", json.dumps(rank_b, ensure_ascii=False))
            prompt = prompt.replace("{positive_actions}", json.dumps(pos_keys, ensure_ascii=False))
            prompt = prompt.replace("{negative_actions}", json.dumps(neg_keys, ensure_ascii=False))
            prompt = prompt.replace("{card_pool}", json.dumps(ctx.get("card_pool", []), ensure_ascii=False))
            prompt = prompt.replace("{context_summary}", context_summary)

            try:
                resp = await client.chat.completions.create(
                    model=model,
                    temperature=0.1,
                    max_tokens=100,
                    messages=[{"role": "user", "content": prompt}],
                    extra_body={"enable_thinking": False},
                )
                content = (resp.choices[0].message.content or "").strip()
                return parse_pairwise_response(content)
            except Exception as e:
                logger.warning("Async pairwise judge failed: %s", e)
                return 0

    tasks = [_judge_one(p) for p in pairs]
    return await asyncio.gather(*tasks)

