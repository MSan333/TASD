"""TipBank 加载与检索 — 自包含模块, 不依赖 rank-agent.

从 rank-agent 中迁移的最小依赖集:
  - Skill 数据类
  - TipIndex BM25 检索引擎
  - generate_search_queries (LLM 辅助 query 生成)

用法:
    from exp.grpo.tipbank_loader import load_tipbank, TipIndex, generate_search_queries

    tips = load_tipbank("/path/to/tipbank")
    tip_index = TipIndex(tips)
    queries = generate_search_queries(llm_client, "qwen-turbo", ctx, card_pool)
    matched = tip_index.search(queries, top_k=15, venture="PH", action_names=card_pool)
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# Skill 数据类
# ============================================================================

class Skill:
    """Tip/Skill 数据结构."""

    def __init__(
        self,
        skill_id: str = "",
        name: str = "",
        category: str = "general",
        principle: str = "",
        when_to_apply: str = "",
        how_to_apply: str = "",
        source_trajectories: Optional[List[str]] = None,
        confidence: float = 0.5,
        effectiveness_score: float = 0.0,
        created_at: Optional[str] = None,
        last_validated: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ):
        self.skill_id = skill_id
        self.name = name
        self.category = category
        self.principle = principle
        self.when_to_apply = when_to_apply
        self.how_to_apply = how_to_apply
        self.source_trajectories = source_trajectories or []
        self.confidence = confidence
        self.effectiveness_score = effectiveness_score
        self.created_at = created_at or datetime.now().isoformat()
        self.last_validated = last_validated
        self.extra = extra or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "category": self.category,
            "principle": self.principle,
            "when_to_apply": self.when_to_apply,
            "how_to_apply": self.how_to_apply,
            "confidence": self.confidence,
            "effectiveness_score": self.effectiveness_score,
            "created_at": self.created_at,
            "last_validated": self.last_validated,
            **self.extra,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Skill":
        reserved_keys = {
            "skill_id", "name", "category", "principle",
            "when_to_apply", "how_to_apply", "source_trajectories",
            "confidence", "effectiveness_score", "created_at", "last_validated",
        }
        return cls(
            skill_id=data.get("skill_id", ""),
            name=data.get("name", ""),
            category=data.get("category", "general"),
            principle=data.get("principle", ""),
            when_to_apply=data.get("when_to_apply", ""),
            how_to_apply=data.get("how_to_apply", ""),
            source_trajectories=data.get("source_trajectories", []),
            confidence=data.get("confidence", 0.5),
            effectiveness_score=data.get("effectiveness_score", 0.0),
            created_at=data.get("created_at"),
            last_validated=data.get("last_validated"),
            extra={k: v for k, v in data.items() if k not in reserved_keys},
        )


# ============================================================================
# TipBank 加载
# ============================================================================

def load_tipbank(tipbank_path: str) -> List[Skill]:
    """加载 TipBank 目录/文件, 返回 Skill 列表."""
    if not os.path.exists(tipbank_path):
        logger.warning("TipBank not found: %s", tipbank_path)
        return []

    if os.path.isdir(tipbank_path):
        return _load_from_directory(tipbank_path)
    else:
        return _load_single_file(tipbank_path)


def _load_from_directory(dir_path: str) -> List[Skill]:
    manifest_path = os.path.join(dir_path, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        files = [
            os.path.join(dir_path, entry["file"])
            for entry in manifest.get("skill_files", [])
            if os.path.exists(os.path.join(dir_path, entry["file"]))
        ]
    else:
        files = sorted(
            os.path.join(dir_path, f)
            for f in os.listdir(dir_path)
            if f.endswith(".json")
            and f not in ("manifest.json", "tipbank_recycle.json")
        )

    all_skills: List[Skill] = []
    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                all_skills.extend(Skill.from_dict(s) for s in data)
            elif isinstance(data, dict):
                for s in data.get("general_skills", []):
                    all_skills.append(Skill.from_dict(s))
                for domain_skills in data.get("domain_skills", {}).values():
                    for s in domain_skills:
                        all_skills.append(Skill.from_dict(s))
                for s in data.get("tips_bank", []):
                    all_skills.append(Skill.from_dict(s))
        except Exception as e:
            logger.warning("Failed to load %s: %s", file_path, e)

    logger.info("TipBank loaded from directory: %d skills from %d files", len(all_skills), len(files))
    return all_skills


def _load_single_file(path: str) -> List[Skill]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning("Failed to load TipBank file %s: %s", path, e)
        return []

    all_skills: List[Skill] = []
    if isinstance(data, list):
        all_skills = [Skill.from_dict(s) for s in data]
    elif isinstance(data, dict):
        for s in data.get("general_skills", []):
            all_skills.append(Skill.from_dict(s))
        for domain_skills in data.get("domain_skills", {}).values():
            for s in domain_skills:
                all_skills.append(Skill.from_dict(s))
        for s in data.get("tips_bank", []):
            all_skills.append(Skill.from_dict(s))

    logger.info("TipBank loaded: %d skills from %s", len(all_skills), path)
    return all_skills


# ============================================================================
# TipIndex — BM25 检索引擎
# ============================================================================

SYNONYM_MAP: Dict[str, List[str]] = {
    "roas": ["ioproas", "roi", "roas"],
    "ioproas": ["roas", "roi", "ioproas"],
    "roi": ["roas", "ioproas", "roi"],
    "预算": ["budget", "预算"],
    "budget": ["预算", "budget"],
    "出价": ["bid", "maxbid", "max_bid", "max_cpc", "出价"],
    "bid": ["出价", "maxbid", "max_bid", "bid"],
    "maxbid": ["出价", "bid", "max_bid", "maxbid"],
    "max_bid": ["出价", "bid", "maxbid", "max_bid"],
    "max_cpc": ["出价", "bid", "max_cpc"],
    "troi": ["troi", "troi目标"],
    "充值": ["topup", "充值", "手动充值"],
    "topup": ["充值", "topup", "手动充值"],
    "余额": ["balance", "余额", "资金"],
    "balance": ["余额", "balance", "资金"],
    "扩量": ["扩量", "放量", "增长", "提量"],
    "收缩": ["收缩", "降参", "控本", "控成本", "限流"],
    "冷启动": ["冷启", "冷启动", "低活", "新建"],
    "新建": ["create", "新建", "创建", "冷启动"],
    "create": ["新建", "create", "创建"],
    "计划": ["campaign", "计划"],
    "campaign": ["计划", "campaign"],
}

_BM25_K1 = 1.5
_BM25_B = 0.75


class TipIndex:
    """BM25 Tip 文本检索引擎, 支持分区检索."""

    def __init__(self, tips: List[Skill]):
        self.tips = list(tips)
        self.tip_texts: List[str] = []
        self.tip_tokens: List[List[str]] = []
        self.tip_ventures: List[Set[str]] = []
        self.tip_actions: List[Set[str]] = []
        self.tip_categories: List[str] = []

        self.inverted_index: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0.0
        self.doc_freq: Dict[str, int] = defaultdict(int)
        self.n_docs: int = len(self.tips)

        self._partition_indices: Dict[str, List[int]] = defaultdict(list)

        if self.tips:
            self._build_index()

    def _build_index(self):
        total_length = 0

        for idx, tip in enumerate(self.tips):
            text = f"{tip.name} {tip.principle} {tip.when_to_apply} {tip.how_to_apply}"
            self.tip_texts.append(text)

            tokens = self._tokenize(text)
            self.tip_tokens.append(tokens)

            doc_len = len(tokens)
            self.doc_lengths.append(doc_len)
            total_length += doc_len

            ventures = set(tip.extra.get('ventures', []))
            if not ventures and tip.category not in ('general', 'tip'):
                ventures.add(tip.category)
            self.tip_ventures.append(ventures)

            actions = set(tip.extra.get('action_names', []))
            self.tip_actions.append(actions)

            category = tip.category
            self.tip_categories.append(category)

            self._partition_indices["__all__"].append(idx)
            if category == "general":
                self._partition_indices["general"].append(idx)
            elif category == "tip":
                self._partition_indices["tip"].append(idx)
            else:
                self._partition_indices[f"domain_{category}"].append(idx)

            tf_counts: Dict[str, int] = defaultdict(int)
            for token in tokens:
                tf_counts[token] += 1

            seen_terms: Set[str] = set()
            for token, count in tf_counts.items():
                tf = count / doc_len if doc_len > 0 else 0
                self.inverted_index[token].append((idx, tf))
                if token not in seen_terms:
                    self.doc_freq[token] += 1
                    seen_terms.add(token)

        self.avg_doc_length = total_length / self.n_docs if self.n_docs > 0 else 1.0

    def search(
        self,
        queries: List[str],
        top_k: int = 5,
        venture: Optional[str] = None,
        action_names: Optional[List[str]] = None,
        min_score: float = 0.0,
    ) -> List[Skill]:
        """多 query 联合检索, 返回 top_k 个最相关的 Tip."""
        if not queries or not self.tips:
            return []

        allowed_docs = self._get_allowed_docs(venture)
        scores: Dict[int, float] = defaultdict(float)

        for query in queries:
            query_tokens = self._tokenize_query(query)
            for token in query_tokens:
                if token not in self.inverted_index:
                    continue
                idf = self._idf(token)
                for doc_idx, tf in self.inverted_index[token]:
                    if allowed_docs is not None and doc_idx not in allowed_docs:
                        continue
                    dl = self.doc_lengths[doc_idx]
                    bm25 = idf * (tf * (_BM25_K1 + 1)) / (
                        tf + _BM25_K1 * (1 - _BM25_B + _BM25_B * dl / self.avg_doc_length)
                    )
                    scores[doc_idx] += bm25

        if venture:
            for doc_idx in scores:
                if venture in self.tip_ventures[doc_idx]:
                    scores[doc_idx] += 0.3

        if action_names:
            action_set = set(action_names)
            for doc_idx in scores:
                overlap = action_set & self.tip_actions[doc_idx]
                if overlap:
                    scores[doc_idx] += 0.5 * len(overlap)

        ranked = sorted(scores.items(), key=lambda x: -x[1])
        results = []
        for doc_idx, score in ranked:
            if score < min_score:
                break
            results.append(self.tips[doc_idx])
            if len(results) >= top_k:
                break

        return results

    def _get_allowed_docs(self, venture: Optional[str]) -> Optional[Set[int]]:
        if not venture:
            return None
        domain_key = f"domain_{venture}"
        if domain_key not in self._partition_indices and venture not in self._partition_indices:
            return None
        allowed = set(self._partition_indices.get("general", []))
        allowed.update(self._partition_indices.get("tip", []))
        allowed.update(self._partition_indices.get(domain_key, []))
        return allowed

    def _idf(self, term: str) -> float:
        df = self.doc_freq.get(term, 0)
        if df == 0:
            return 0.0
        return math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)

    def _tokenize(self, text: str) -> List[str]:
        tokens: List[str] = []
        for w in re.findall(r'[a-zA-Z_]\w{1,}', text):
            tokens.append(w.lower())
        chinese_segments = re.findall(r'[一-鿿]+', text)
        for seg in chinese_segments:
            tokens.append(seg)
            for i in range(len(seg) - 1):
                tokens.append(seg[i:i + 2])
            if len(seg) >= 3:
                for i in range(len(seg) - 2):
                    tokens.append(seg[i:i + 3])
        return tokens

    def _tokenize_query(self, query: str) -> Set[str]:
        raw_tokens = set(self._tokenize(query))
        expanded: Set[str] = set(raw_tokens)
        for token in raw_tokens:
            synonyms = SYNONYM_MAP.get(token, [])
            expanded.update(synonyms)
        return expanded

    def get_stats(self) -> Dict[str, Any]:
        partitions = {
            k: len(v) for k, v in self._partition_indices.items() if k != "__all__"
        }
        return {
            "total_tips": self.n_docs,
            "vocab_size": len(self.inverted_index),
            "avg_doc_length": round(self.avg_doc_length, 1),
            "partitions": partitions,
        }


# ============================================================================
# Query 生成 (LLM 辅助)
# ============================================================================

_QUERY_GEN_PROMPT = """\
你是广告排序策略检索助手。根据以下商家状态，生成 3-5 个检索短语，用于从策略库中检索最相关的排序策略。

要求：
- 每个短语 3-8 个字，聚焦一个具体场景或信号
- 覆盖：行为意图、资金状态、计划结构、卡片类型等维度
- 直接输出 JSON 数组，不要任何其他内容

商家状态：
- venture: {venture}
- 近期行为: {behavior_summary}
- 卡片池: {card_pool}
- 钱包: balance={balance}, l7d_spend={l7d_spend}
- 计划数: {campaign_count}
- 商家随后执行了: {positive_actions}"""


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def generate_search_queries(
    llm_client,
    query_model: str,
    ctx: Dict[str, Any],
    card_pool: List[str],
    extra_context: str = "",
) -> List[str]:
    """LLM 生成检索 query, 失败时 fallback 到关键词提取."""
    wallet_raw = ctx.get("wallet_data", {})
    if isinstance(wallet_raw, str):
        try:
            wallet_raw = json.loads(wallet_raw)
        except (json.JSONDecodeError, TypeError):
            wallet_raw = {}
    wallet_data = wallet_raw if isinstance(wallet_raw, dict) else {}
    balance = _safe_float(wallet_data.get("balance", 0))
    l7d_spend = _safe_float(wallet_data.get("l7d_spend", 0))

    camp_raw = ctx.get("campaign_data", {})
    if isinstance(camp_raw, str):
        try:
            camp_raw = json.loads(camp_raw)
        except (json.JSONDecodeError, TypeError):
            camp_raw = {}
    camp_data = camp_raw if isinstance(camp_raw, dict) else {}
    summary = camp_data.get("summary", {}) if isinstance(camp_data, dict) else {}
    campaign_count = int(summary.get("total_count", 0)) if summary else 0

    seller_behavior = str(ctx.get("seller_behavior", ""))
    behavior_summary = seller_behavior[:300] if seller_behavior else "无"

    positive_actions = ctx.get("positive_actions", [])
    if isinstance(positive_actions, list):
        positive_actions = ", ".join(positive_actions)

    prompt = _QUERY_GEN_PROMPT.format(
        venture=ctx.get("venture", "未知"),
        behavior_summary=behavior_summary,
        card_pool=", ".join(card_pool[:15]),
        balance=f"{balance:.0f}",
        l7d_spend=f"{l7d_spend:.0f}",
        campaign_count=campaign_count,
        positive_actions=positive_actions or extra_context or "未知",
    )

    try:
        resp = llm_client.chat.completions.create(
            model=query_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200,
            timeout=15,
        )
        content = (resp.choices[0].message.content or "").strip()
        match = re.search(r'\[.*\]', content, re.DOTALL)
        if match:
            queries = json.loads(match.group())
            if isinstance(queries, list) and queries:
                return [str(q) for q in queries[:5]]
    except Exception as e:
        logger.warning("Query generation failed: %s", e)

    return _fallback_queries(seller_behavior, card_pool)


def _fallback_queries(seller_behavior: str, card_pool: List[str]) -> List[str]:
    queries: List[str] = []
    behavior_lower = seller_behavior.lower()

    if "create" in behavior_lower:
        queries.append("新建计划扩量")
    if "ioproas" in behavior_lower or "roas" in behavior_lower:
        queries.append("ROAS调整优化")
    if "topup" in behavior_lower:
        queries.append("充值后资金利用")
    if "budget" in behavior_lower:
        queries.append("预算调整策略")
    if "off" in behavior_lower or "remove" in behavior_lower:
        queries.append("关停收缩意图")

    pool_str = " ".join(card_pool).lower()
    if "create" in pool_str:
        queries.append("冷启动建计划")
    if "topup" in pool_str:
        queries.append("余额资金风险")

    if not queries:
        queries = ["排序通用策略"]

    return queries[:5]
