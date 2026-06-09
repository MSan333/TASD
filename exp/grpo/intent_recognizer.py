"""商家意图识别模块.

从商家真实行为中归纳意图，而非猜测。
集成 skills/analysis 下的引导分析 Skill，结构化分析后再输出意图。

输入: seller_behavior + campaign_data + wallet_data + merchant_memory
输出: primary_intent + confidence + evidence + risk_flags + recommended_direction
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ============================================================================
# Skills 加载 (进程级单例)
# ============================================================================

_SKILL_LOADER = None


def _get_skill_loader():
    global _SKILL_LOADER
    if _SKILL_LOADER is not None:
        return _SKILL_LOADER

    skills_dir = os.path.join(os.path.dirname(__file__), "skills")
    if not os.path.isdir(skills_dir):
        logger.debug("Skills dir not found: %s", skills_dir)
        return None

    try:
        from .skills import SkillLoader
        _SKILL_LOADER = SkillLoader(skills_dir)
        logger.info("Analysis skills loaded for intent recognition")
        return _SKILL_LOADER
    except Exception as e:
        logger.warning("Failed to load skills: %s", e)
        return None

INTENT_PROMPT = """\
你是广告商家意图分析专家。根据以下信息，识别商家的真实意图。

{analysis_skills}

## 商家近期行为日志
{seller_behavior}

## 行为方向摘要
{behavior_direction_summary}

## 钱包数据
{wallet_data}

## 广告投放数据
{campaign_data}

## 商家历史校准（如有）
{merchant_memory}

请先按照上述分析框架逐项分析，然后输出 JSON（不要输出其他内容）:
{
  "primary_intent": "扩量 | 控成本 | 新建计划 | 维持现状 | 收缩 | 充值续命",
  "intent_confidence": 0.0-1.0,
  "intent_evidence": ["证据1", "证据2"],
  "risk_flags": [],
  "recommended_direction": "应该推什么类型的卡片排在前面",
  "should_avoid": "什么类型的卡片不该排前面"
}
"""

DEFAULT_INTENT = {
    "primary_intent": "维持现状",
    "intent_confidence": 0.0,
    "intent_evidence": [],
    "risk_flags": [],
    "recommended_direction": "",
    "should_avoid": "",
}


def _summarize_behavior_direction(seller_behavior: str) -> str:
    """从行为日志提取方向摘要."""
    if not seller_behavior:
        return "无近期行为"

    directions = []
    behavior_lower = seller_behavior.lower()

    budget_up = behavior_lower.count("budget from") - behavior_lower.count("budget from -1")
    if budget_up > 0:
        directions.append(f"提升预算 {budget_up} 次")

    roas_changes = len(re.findall(r'ioproas from (\d+\.?\d*) to (\d+\.?\d*)', behavior_lower))
    if roas_changes:
        directions.append(f"调整 ROAS {roas_changes} 次")

    if "remove campaign" in behavior_lower:
        directions.append("关闭/删除计划")

    if "rapidboost from on to off" in behavior_lower:
        directions.append("关闭极速起量（控成本倾向）")
    elif "rapidboost from off to on" in behavior_lower:
        directions.append("开启极速起量（扩量倾向）")

    if "create" in behavior_lower:
        directions.append("新建计划")

    if "topup" in behavior_lower:
        directions.append("充值操作")

    return "; ".join(directions) if directions else "行为方向不明确"


def recognize_intent(
    ctx: Dict[str, Any],
    client,
    model: str,
    merchant_memory: Optional[str] = None,
) -> Dict[str, Any]:
    """识别商家意图.

    Args:
        ctx: 包含 seller_behavior, wallet_data, campaign_data 的上下文
        client: OpenAI 兼容客户端
        model: 模型名
        merchant_memory: 商家历史校准记忆（可选）

    Returns:
        意图识别结果 dict
    """
    seller_behavior = str(ctx.get("seller_behavior", ""))[:2000]
    behavior_direction = _summarize_behavior_direction(seller_behavior)

    wallet_data = ctx.get("wallet_data", "")
    if isinstance(wallet_data, dict):
        wallet_data = json.dumps(wallet_data, ensure_ascii=False)

    campaign_data = ctx.get("campaign_data", "")
    if isinstance(campaign_data, dict):
        campaign_data = json.dumps(campaign_data, ensure_ascii=False)

    # 加载 analysis skills 作为分析框架注入 prompt
    analysis_skills_text = ""
    skill_loader = _get_skill_loader()
    if skill_loader:
        skill_ctx = {
            "seller_behavior": seller_behavior,
            "wallet_data": wallet_data,
            "campaign_data": campaign_data,
            "venture": ctx.get("venture", ""),
            "card_pool": ctx.get("card_pool", []),
        }
        skills_prompt, selected = skill_loader.build_analysis_prompt(skill_ctx)
        if skills_prompt:
            analysis_skills_text = f"## 分析框架（请逐项完成）\n\n{skills_prompt}"
            logger.debug("Intent recognition using skills: %s", selected)

    # 手动替换避免 skill 内容中的 {} 干扰 .format()
    prompt = INTENT_PROMPT.replace("{analysis_skills}", analysis_skills_text)
    prompt = prompt.replace("{seller_behavior}", seller_behavior[:1500])
    prompt = prompt.replace("{behavior_direction_summary}", behavior_direction)
    prompt = prompt.replace("{wallet_data}", str(wallet_data)[:500])
    prompt = prompt.replace("{campaign_data}", str(campaign_data)[:1000])
    prompt = prompt.replace("{merchant_memory}", merchant_memory or "（无历史校准记录）")

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
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            result = json.loads(match.group())
            result.setdefault("primary_intent", "维持现状")
            result.setdefault("intent_confidence", 0.5)
            result.setdefault("intent_evidence", [])
            result.setdefault("risk_flags", [])
            return result
    except Exception as e:
        logger.warning("Intent recognition failed: %s", e)

    return dict(DEFAULT_INTENT)
