#!/usr/bin/env python3
# Simple PyTorch Dataset for next-word prediction using sliding windows.

import os, json
import numpy as np
import torch
from torch.utils.data import Dataset

DATA_DIR = "data"

class NextWordDataset(Dataset):
    def __init__(self, dataset_name="warpeace", context_len=5, split="train", split_ratio=0.9):
        assert split in ("train","val","test")
        ds_dir = os.path.join(DATA_DIR, dataset_name)
        with open(os.path.join(ds_dir, "vocab.json"), "r", encoding="utf-8") as f:
            info = json.load(f)
        self.vocab = info["vocab"]
        self.word2idx = {w:i for i,w in enumerate(self.vocab)}
        tokens_path = os.path.join(ds_dir, "tokens.ids")
        with open(tokens_path, "r", encoding="utf-8") as f:
            tokens = list(map(int, f.read().strip().split()))
        X, y = [], []
        for i in range(context_len, len(tokens)):
            X.append(tokens[i-context_len:i])
            y.append(tokens[i])
        X = np.stack(X)
        y = np.array(y, dtype=np.int64)
        # deterministic split
        split_idx = int(len(X)*split_ratio)
        if split == "train":
            sel = slice(0, split_idx)
        else:
            sel = slice(split_idx, None)
        self.X = torch.from_numpy(X[sel]).long()
        self.y = torch.from_numpy(y[sel]).long()
        self.context_len = context_len
        print(f"{dataset_name} {split}: {len(self.X)} examples, vocab_size={len(self.vocab)}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
