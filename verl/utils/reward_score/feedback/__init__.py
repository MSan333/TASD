from verl.utils.reward_score.feedback import math
from verl.utils.reward_score.feedback import code
from verl.utils.reward_score.feedback import gpqa
from verl.utils.reward_score.feedback import mcq
from verl.utils.reward_score.feedback import tooluse


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
) -> dict:
    """Single-sample reward (used by NaiveRewardManager)."""
    if data_source in ["code", "livecodebench", "humanevalplus"]:
        results = code.compute_score(solution_str, ground_truth, extra_info, sparse_rewards=True, max_test_cases=None)
    elif data_source in ["math", "math500", "dapo_math", "gsm8k"]:
        results = math.compute_score(solution_str, ground_truth, extra_info)
    elif data_source in ["gpqa"]:
        results = gpqa.compute_score(solution_str, ground_truth)
    elif data_source in ["sciknoweval"]:
        results = mcq.compute_score(solution_str, ground_truth)
    elif data_source in ["tooluse"]:
        results = tooluse.compute_score(solution_str, ground_truth)
    elif data_source in ["ranking"]:
        from exp.grpo.ranking import compute_score as ranking_compute_score
        results = ranking_compute_score(solution_str, ground_truth, extra_info)
    else:
        raise ValueError(f"Reward style {data_source} not found.")
    return results


def compute_score_batch(
    data_sources: list,
    solution_strs: list,
    ground_truths: list,
    extra_infos: list = None,
    **kwargs,
) -> list:
    """Batch reward (used by BatchRewardManager).

    For 'ranking' data_source, uses batch LLM Judge for efficiency.
    For other data sources, falls back to per-sample computation.
    """
    if extra_infos is None:
        extra_infos = [None] * len(data_sources)

    # Check if all samples are ranking — use batch mode
    if all(ds == "ranking" for ds in data_sources):
        from exp.grpo.ranking_batch import compute_score as batch_ranking_score
        return batch_ranking_score(
            data_sources=data_sources,
            solution_strs=solution_strs,
            ground_truths=ground_truths,
            extra_infos=extra_infos,
            **kwargs,
        )

    # Mixed data sources — fall back to per-sample
    results = []
    for i in range(len(data_sources)):
        result = compute_score(
            data_source=data_sources[i],
            solution_str=solution_strs[i],
            ground_truth=ground_truths[i],
            extra_info=extra_infos[i],
        )
        results.append(result)
    return results