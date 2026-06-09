"""Knowledge-Grounded Judge — 知识驱动排序质量评分.

三步串行:
  Step 1: TipBank 检索 (复用 tipbank_loader)
  Step 2: 商家意图识别 (复用 intent_recognizer)
  Step 3: 知识对比打分 — LLM 结合 tips + intent + matching skill + rank_list 综合评判

评判维度:
  A. 意图对齐 (40%) — 排序与商家真实意图是否一致
  B. 策略合规 (30%) — 排序是否符合 TipBank 专家策略
  C. 结果预测 (20%) — 排序与商家实际行为的吻合度
  D. 风险规避 (10%) — 是否避免了危险推荐
"""

from __future__ import annotations

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
你是排序质量评判专家。请基于以下信息，评判模型的排序结果。

## 1. 相关排序策略（从专家知识库检索）
{retrieved_tips}

## 2. 商家意图识别结果
{intent_result}

## 3. 卡片-意图匹配准则
{matching_skill}

## 4. 模型的排序输出
{rank_list}

## 5. 商家实际后续行为（ground truth）
- 正向操作（商家确实做了）: {positive_actions}
- 反向操作（商家做了相反的）: {negative_actions}

## 6. 候选卡片池
{card_pool}

## 评判维度

### A. 意图对齐 (40%)
模型排序是否与识别出的商家意图一致？
- Top-1 卡片与 primary_intent 方向一致 → 高分
- Top-3 中有与意图冲突的卡片 → 扣分
- 排序完全违背意图 → 严重扣分

### B. 策略合规 (30%)
模型排序是否符合检索到的专家策略？
- 排序行为与策略的 how_to_apply 一致 → 加分
- 排序触发了策略的 when_to_apply 但没遵循 → 扣分
- 排序触犯了防御型策略的约束 → 严重扣分

### C. 结果预测 (20%)
模型排序与商家实际行为的吻合度：
- 正向操作排在 Top-3 → 加分
- 负向操作排在 Top-3 → 扣分
（注意：商家行为本身不一定最优，需结合意图和策略综合判断）

### D. 风险规避 (10%)
- 资金风险下推了花钱卡 → 严重扣分
- 无活跃计划时推了操作卡 → 扣分

请直接输出 JSON（不要输出其他内容）:
{
  "score": -1.0 到 1.0,
  "intent_alignment": 0.0-1.0,
  "strategy_compliance": 0.0-1.0,
  "result_prediction": 0.0-1.0,
  "risk_avoidance": 0.0-1.0,
  "reasoning": "一句话总结"
}
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
            for key in ("intent_alignment", "strategy_compliance", "result_prediction", "risk_avoidance"):
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


def knowledge_judge_single(
    rank_list: List[str],
    ctx: Dict[str, Any],
    tips: list,
    intent: Dict[str, Any],
    positive_keys: List[str],
    negative_keys: List[str],
    client,
    model: str,
) -> Dict[str, Any]:
    """对单个 rank_list 执行知识驱动打分.

    Args:
        rank_list: 模型输出的排序列表
        ctx: 样本上下文
        tips: 检索到的 TipBank 策略
        intent: 意图识别结果
        positive_keys: 正向动作标签
        negative_keys: 负向动作标签
        client: OpenAI 兼容客户端
        model: Judge 模型名

    Returns:
        评分结果 dict, 包含 score + 各维度分数 + reasoning
    """
    card_pool = ctx.get("card_pool", rank_list)

    matching_skill = _get_matching_prompt() or "（无匹配准则）"

    # 手动替换避免 skill/tips 内容中的 {} 干扰 .format()
    prompt = JUDGE_PROMPT.replace("{retrieved_tips}", format_tips_for_judge(tips))
    prompt = prompt.replace("{intent_result}", json.dumps(intent, ensure_ascii=False))
    prompt = prompt.replace("{matching_skill}", matching_skill)
    prompt = prompt.replace("{rank_list}", json.dumps(rank_list[:8], ensure_ascii=False))
    prompt = prompt.replace("{positive_actions}", json.dumps(positive_keys, ensure_ascii=False))
    prompt = prompt.replace("{negative_actions}", json.dumps(negative_keys, ensure_ascii=False))
    prompt = prompt.replace("{card_pool}", json.dumps(card_pool[:12], ensure_ascii=False))

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
