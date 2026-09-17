"""
PyTorch Dataset and DataLoader for Vaayu SLMM training.
Handles tokenization, sequence truncation/packing, and label formatting.
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

class VaayuTextDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer: PreTrainedTokenizerFast, max_length: int = 1024):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        path = Path(jsonl_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file {path} not found.")

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    self.samples.append(item.get("text", ""))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.samples[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # In causal LM, labels are identical to input_ids except padded positions are masked with -100
        labels = input_ids.clone()
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        labels[labels == pad_id] = -100

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }
