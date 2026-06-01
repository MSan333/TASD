"""
Preprocess preference datasets into DPO-ready parquet format.

Input: HuggingFace preference datasets with prompt/chosen/rejected columns.
Output: parquet files with columns: prompt (str), chosen (str), rejected (str).

Usage:
    # From a HuggingFace dataset
    python data/preprocess_dpo.py \
        --dataset_name Anthropic/hh-rlhf \
        --output_dir datasets/hh_rlhf_dpo

    # From a local jsonl file
    python data/preprocess_dpo.py \
        --input_file data/my_preferences.jsonl \
        --output_dir datasets/my_dpo_data

    # With custom column names
    python data/preprocess_dpo.py \
        --dataset_name my_org/my_dataset \
        --prompt_key question \
        --chosen_key preferred \
        --rejected_key dispreferred \
        --output_dir datasets/my_dpo_data
"""

import argparse
import os

import pandas as pd


def load_hf_dataset(dataset_name: str, split: str = None):
    """Load a HuggingFace dataset."""
    from datasets import load_dataset

    if split:
        return load_dataset(dataset_name, split=split)
    return load_dataset(dataset_name)


def process_hh_rlhf(dataset_name: str, output_dir: str, test_ratio: float = 0.1, seed: int = 42):
    """Process Anthropic/hh-rlhf style datasets (chosen/rejected are full conversations)."""
    from datasets import load_dataset

    ds = load_dataset(dataset_name)

    def extract_last_turn(conversation: str) -> tuple[str, str]:
        """Extract prompt and response from HH-RLHF format."""
        parts = conversation.split("\n\nAssistant: ")
        if len(parts) < 2:
            return conversation, ""
        response = parts[-1]
        prompt = "\n\nAssistant: ".join(parts[:-1])
        if prompt.startswith("\n\nHuman: "):
            prompt = prompt[len("\n\nHuman: "):]
        return prompt, response

    records = []
    for split_name in ds:
        for example in ds[split_name]:
            chosen_text = example.get("chosen", "")
            rejected_text = example.get("rejected", "")
            chosen_prompt, chosen_response = extract_last_turn(chosen_text)
            rejected_prompt, rejected_response = extract_last_turn(rejected_text)
            if chosen_prompt == rejected_prompt and chosen_response and rejected_response:
                records.append({
                    "prompt": chosen_prompt,
                    "chosen": chosen_response,
                    "rejected": rejected_response,
                })

    df = pd.DataFrame(records)
    _save_splits(df, output_dir, test_ratio, seed)


def process_standard(
    dataset_name: str = None,
    input_file: str = None,
    output_dir: str = "datasets/dpo",
    prompt_key: str = "prompt",
    chosen_key: str = "chosen",
    rejected_key: str = "rejected",
    test_ratio: float = 0.1,
    seed: int = 42,
):
    """Process datasets that already have prompt/chosen/rejected columns."""
    if input_file:
        if input_file.endswith(".jsonl") or input_file.endswith(".json"):
            df = pd.read_json(input_file, lines=input_file.endswith(".jsonl"))
        elif input_file.endswith(".parquet"):
            df = pd.read_parquet(input_file)
        else:
            raise ValueError(f"Unsupported file format: {input_file}")
    else:
        from datasets import load_dataset
        ds = load_dataset(dataset_name, split="train")
        df = ds.to_pandas()

    # Rename columns to standard names
    df = df.rename(columns={
        prompt_key: "prompt",
        chosen_key: "chosen",
        rejected_key: "rejected",
    })

    # Keep only required columns
    df = df[["prompt", "chosen", "rejected"]].dropna()

    _save_splits(df, output_dir, test_ratio, seed)


def _save_splits(df: pd.DataFrame, output_dir: str, test_ratio: float, seed: int):
    """Split dataframe into train/test and save as parquet."""
    os.makedirs(output_dir, exist_ok=True)

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n_test = max(1, int(len(df) * test_ratio))
    test_df = df.iloc[:n_test]
    train_df = df.iloc[n_test:]

    train_path = os.path.join(output_dir, "train.parquet")
    test_path = os.path.join(output_dir, "test.parquet")

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print(f"Saved {len(train_df)} train samples to {train_path}")
    print(f"Saved {len(test_df)} test samples to {test_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess preference data for DPO training")
    parser.add_argument("--dataset_name", type=str, default=None, help="HuggingFace dataset name")
    parser.add_argument("--input_file", type=str, default=None, help="Local input file (json/jsonl/parquet)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for parquet files")
    parser.add_argument("--prompt_key", type=str, default="prompt", help="Column name for prompt")
    parser.add_argument("--chosen_key", type=str, default="chosen", help="Column name for chosen response")
    parser.add_argument("--rejected_key", type=str, default="rejected", help="Column name for rejected response")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--format", type=str, default="standard", choices=["standard", "hh_rlhf"],
                        help="Dataset format: 'standard' (prompt/chosen/rejected) or 'hh_rlhf' (Anthropic format)")

    args = parser.parse_args()

    if args.format == "hh_rlhf":
        assert args.dataset_name, "--dataset_name required for hh_rlhf format"
        process_hh_rlhf(args.dataset_name, args.output_dir, args.test_ratio, args.seed)
    else:
        process_standard(
            dataset_name=args.dataset_name,
            input_file=args.input_file,
            output_dir=args.output_dir,
            prompt_key=args.prompt_key,
            chosen_key=args.chosen_key,
            rejected_key=args.rejected_key,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
