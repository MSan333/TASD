#!/usr/bin/env python3
"""Parse training_output.log and plot key metrics."""

import re
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

LOG_FILE = Path(__file__).parent / "training_output.log"
OUTPUT_DIR = Path(__file__).parent / "plots"


def parse_log(log_path: str = None):
    """Parse training log and extract metrics per step."""
    path = Path(log_path) if log_path else LOG_FILE
    text = path.read_text()

    steps = []
    batch_rewards = []

    # Parse step metrics (format: step:N - key:value - key:value ...)
    step_pattern = re.compile(
        r"step:(\d+)\s*-\s*(.*?)(?=\n|$)"
    )
    for match in step_pattern.finditer(text):
        step_num = int(match.group(1))
        metrics_str = match.group(2)
        metrics = {}
        for pair in metrics_str.split(" - "):
            if ":" in pair:
                key, value = pair.split(":", 1)
                key = key.strip()
                value = value.strip()
                # Extract numeric value
                value = re.sub(r"np\.\w+\((.*?)\)", r"\1", value)
                try:
                    metrics[key] = float(value)
                except ValueError:
                    pass
        metrics["step"] = step_num
        steps.append(metrics)

    # Parse BatchReward lines
    batch_pattern = re.compile(
        r"\[BatchReward\] samples=(\d+), format_valid=(\d+), "
        r"llm_judged=(\d+), avg_score=([-\d.]+), avg_llm=([-\d.]+)"
    )
    for match in batch_pattern.finditer(text):
        batch_rewards.append({
            "samples": int(match.group(1)),
            "format_valid": int(match.group(2)),
            "llm_judged": int(match.group(3)),
            "avg_score": float(match.group(4)),
            "avg_llm": float(match.group(5)),
        })

    return steps, batch_rewards


def plot_metrics(steps, batch_rewards, output_dir=None):
    """Generate training metric plots."""
    out = Path(output_dir) if output_dir else OUTPUT_DIR
    out.mkdir(exist_ok=True)

    if not steps:
        print("No step data found in log.")
        return

    step_nums = [s["step"] for s in steps]

    # --- Figure 1: Reward Scores ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("GRPO Training Metrics", fontsize=14, fontweight="bold")

    # 1a: Batch Reward avg_score
    if batch_rewards:
        ax = axes[0, 0]
        scores = [b["avg_score"] for b in batch_rewards]
        ax.plot(range(1, len(scores) + 1), scores, "b-o", markersize=3)
        ax.set_title("Avg Reward Score per Batch")
        ax.set_xlabel("Batch")
        ax.set_ylabel("avg_score")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # 1b: LLM Judge score
    if batch_rewards:
        ax = axes[0, 1]
        llm_scores = [b["avg_llm"] for b in batch_rewards]
        ax.plot(range(1, len(llm_scores) + 1), llm_scores, "r-o", markersize=3)
        ax.set_title("Avg LLM Judge Score per Batch")
        ax.set_xlabel("Batch")
        ax.set_ylabel("avg_llm")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # 1c: Format valid ratio
    if batch_rewards:
        ax = axes[1, 0]
        ratios = [b["format_valid"] / b["samples"] for b in batch_rewards]
        ax.plot(range(1, len(ratios) + 1), ratios, "g-o", markersize=3)
        ax.set_title("Format Valid Ratio")
        ax.set_xlabel("Batch")
        ax.set_ylabel("Ratio")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

    # 1d: critic/score/mean (from step data)
    score_means = [s.get("critic/score/mean", None) for s in steps]
    if any(v is not None for v in score_means):
        ax = axes[1, 1]
        valid = [(n, v) for n, v in zip(step_nums, score_means) if v is not None]
        ax.plot([x[0] for x in valid], [x[1] for x in valid], "m-o", markersize=3)
        ax.set_title("Critic Score Mean per Step")
        ax.set_xlabel("Step")
        ax.set_ylabel("Score")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out / "reward_metrics.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {out / 'reward_metrics.png'}")
    plt.close()

    # --- Figure 2: Training Dynamics ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training Dynamics", fontsize=14, fontweight="bold")

    # 2a: Actor entropy
    entropy = [s.get("actor/entropy", None) for s in steps]
    if any(v is not None for v in entropy):
        ax = axes[0, 0]
        valid = [(n, v) for n, v in zip(step_nums, entropy) if v is not None]
        ax.plot([x[0] for x in valid], [x[1] for x in valid], "b-o", markersize=3)
        ax.set_title("Actor Entropy")
        ax.set_xlabel("Step")
        ax.set_ylabel("Entropy")
        ax.grid(True, alpha=0.3)

    # 2b: KL divergence
    kl = [s.get("rollout_corr/kl", None) for s in steps]
    if any(v is not None for v in kl):
        ax = axes[0, 1]
        valid = [(n, v) for n, v in zip(step_nums, kl) if v is not None]
        ax.plot([x[0] for x in valid], [x[1] for x in valid], "r-o", markersize=3)
        ax.set_title("KL Divergence")
        ax.set_xlabel("Step")
        ax.set_ylabel("KL")
        ax.grid(True, alpha=0.3)

    # 2c: Actor grad norm
    grad_norm = [s.get("actor/grad_norm", None) for s in steps]
    if any(v is not None for v in grad_norm):
        ax = axes[1, 0]
        valid = [(n, v) for n, v in zip(step_nums, grad_norm) if v is not None]
        ax.plot([x[0] for x in valid], [x[1] for x in valid], "g-o", markersize=3)
        ax.set_title("Actor Grad Norm")
        ax.set_xlabel("Step")
        ax.set_ylabel("Grad Norm")
        ax.grid(True, alpha=0.3)

    # 2d: Response length mean
    resp_len = [s.get("response_length/mean", None) for s in steps]
    if any(v is not None for v in resp_len):
        ax = axes[1, 1]
        valid = [(n, v) for n, v in zip(step_nums, resp_len) if v is not None]
        ax.plot([x[0] for x in valid], [x[1] for x in valid], "orange", marker="o", markersize=3)
        ax.set_title("Response Length (mean)")
        ax.set_xlabel("Step")
        ax.set_ylabel("Tokens")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out / "training_dynamics.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {out / 'training_dynamics.png'}")
    plt.close()

    # --- Figure 3: Performance ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Performance", fontsize=14, fontweight="bold")

    # 3a: Time per step
    time_step = [s.get("perf/time_per_step", None) for s in steps]
    if any(v is not None for v in time_step):
        ax = axes[0]
        valid = [(n, v) for n, v in zip(step_nums, time_step) if v is not None]
        ax.plot([x[0] for x in valid], [x[1] for x in valid], "b-o", markersize=3)
        ax.set_title("Time per Step (seconds)")
        ax.set_xlabel("Step")
        ax.set_ylabel("Seconds")
        ax.grid(True, alpha=0.3)

    # 3b: Throughput
    throughput = [s.get("perf/throughput", None) for s in steps]
    if any(v is not None for v in throughput):
        ax = axes[1]
        valid = [(n, v) for n, v in zip(step_nums, throughput) if v is not None]
        ax.plot([x[0] for x in valid], [x[1] for x in valid], "g-o", markersize=3)
        ax.set_title("Throughput (tokens/s)")
        ax.set_xlabel("Step")
        ax.set_ylabel("Tokens/s")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out / "performance.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {out / 'performance.png'}")
    plt.close()

    # Summary table
    print("\n" + "=" * 60)
    print(f"{'Step':>5} | {'Score':>8} | {'LLM':>8} | {'Entropy':>8} | {'KL':>10} | {'Time(s)':>8}")
    print("-" * 60)
    for i, s in enumerate(steps):
        batch_score = batch_rewards[i]["avg_score"] if i < len(batch_rewards) else "N/A"
        batch_llm = batch_rewards[i]["avg_llm"] if i < len(batch_rewards) else "N/A"
        ent = s.get("actor/entropy", "N/A")
        kl_val = s.get("rollout_corr/kl", "N/A")
        time_val = s.get("perf/time_per_step", "N/A")
        print(
            f"{s['step']:>5} | "
            f"{batch_score if isinstance(batch_score, str) else batch_score:>8.4f} | "
            f"{batch_llm if isinstance(batch_llm, str) else batch_llm:>8.4f} | "
            f"{ent if isinstance(ent, str) else ent:>8.4f} | "
            f"{kl_val if isinstance(kl_val, str) else kl_val:>10.6f} | "
            f"{time_val if isinstance(time_val, str) else time_val:>8.1f}"
        )
    print("=" * 60)


if __name__ == "__main__":
    log_path = sys.argv[1] if len(sys.argv) > 1 else None
    steps, batch_rewards = parse_log(log_path)
    print(f"Parsed {len(steps)} steps, {len(batch_rewards)} batch rewards")
    plot_metrics(steps, batch_rewards)
