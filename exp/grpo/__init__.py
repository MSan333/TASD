"""Knowledge-Grounded GRPO for Ranking.

veRL 集成入口:
  NaiveRewardManager → feedback/__init__.py → exp.grpo.ranking.compute_score
  BatchRewardManager → feedback/__init__.py → exp.grpo.ranking_batch.compute_score

Reward 架构 (两层):
  L0: Format Gate — 格式校验, 不合法返回 -2.0
  L2: Knowledge-Grounded Judge — TipBank 检索 + 意图识别 + 知识对比打分
"""
