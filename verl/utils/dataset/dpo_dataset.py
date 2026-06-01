import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from verl.utils import hf_tokenizer


class DPODataset(Dataset):
    """Dataset for DPO training with prompt/chosen/rejected string columns.

    Reads parquet files with columns: prompt (str), chosen (str), rejected (str).
    Tokenizes prompt+chosen and prompt+rejected, masks prompt tokens in labels.
    """

    def __init__(
        self,
        parquet_files: str | list[str],
        tokenizer,
        prompt_key: str = "prompt",
        chosen_key: str = "chosen",
        rejected_key: str = "rejected",
        max_length: int = 2048,
        max_prompt_length: int = 1024,
        cache_dir: str = "~/.cache/verl/dpo",
        max_samples: int = -1,
        shuffle: bool = False,
        seed: Optional[int] = None,
    ):
        if not isinstance(parquet_files, list):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.max_samples = max_samples
        self.shuffle = shuffle
        self.seed = seed
        self.cache_dir = os.path.expanduser(cache_dir)

        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer = tokenizer

        self.prompt_key = prompt_key
        self.chosen_key = chosen_key
        self.rejected_key = rejected_key

        self._download()
        self._read_files()

    def _download(self):
        from verl.utils.fs import copy, is_non_local

        os.makedirs(self.cache_dir, exist_ok=True)
        for i, parquet_file in enumerate(self.parquet_files):
            if is_non_local(parquet_file):
                dst = os.path.join(self.cache_dir, os.path.basename(parquet_file))
                if not os.path.exists(dst):
                    copy(src=parquet_file, dst=dst)
                self.parquet_files[i] = dst

    def _read_files(self):
        dataframes = []
        for parquet_file in self.parquet_files:
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes, ignore_index=True)

        total = len(self.dataframe)
        print(f"[DPODataset] total samples: {total}")

        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rng = np.random.default_rng(self.seed)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.iloc[indices.tolist()].reset_index(drop=True)
            print(f"[DPODataset] selected {self.max_samples} samples")

        self.prompts = self.dataframe[self.prompt_key].tolist()
        self.chosen_responses = self.dataframe[self.chosen_key].tolist()
        self.rejected_responses = self.dataframe[self.rejected_key].tolist()

    def __len__(self):
        return len(self.prompts)

    def _tokenize_pair(self, prompt: str, response: str):
        """Tokenize prompt+response, return input_ids, attention_mask, labels (prompt masked)."""
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        if len(prompt_ids) > self.max_prompt_length:
            prompt_ids = prompt_ids[:self.max_prompt_length]

        response_ids = self.tokenizer.encode(response, add_special_tokens=False)
        if self.tokenizer.eos_token_id is not None:
            response_ids = response_ids + [self.tokenizer.eos_token_id]

        input_ids = prompt_ids + response_ids
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]

        prompt_len = len(prompt_ids)
        seq_len = len(input_ids)

        # Labels: -100 for prompt tokens, actual ids for response tokens
        labels = [-100] * prompt_len + input_ids[prompt_len:]

        # Pad to max_length
        pad_len = self.max_length - seq_len
        attention_mask = [1] * seq_len + [0] * pad_len
        input_ids = input_ids + [self.tokenizer.pad_token_id or 0] * pad_len
        labels = labels + [-100] * pad_len

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        chosen = self.chosen_responses[idx]
        rejected = self.rejected_responses[idx]

        chosen_input_ids, chosen_attention_mask, chosen_labels = self._tokenize_pair(prompt, chosen)
        rejected_input_ids, rejected_attention_mask, rejected_labels = self._tokenize_pair(prompt, rejected)

        return {
            "chosen_input_ids": chosen_input_ids,
            "chosen_attention_mask": chosen_attention_mask,
            "chosen_labels": chosen_labels,
            "rejected_input_ids": rejected_input_ids,
            "rejected_attention_mask": rejected_attention_mask,
            "rejected_labels": rejected_labels,
        }
