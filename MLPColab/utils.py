#!/usr/bin/env python3


import torch, os
import matplotlib.pyplot as plt

def save_checkpoint(state, filename="checkpoint.pth"):
    torch.save(state, filename)

def load_checkpoint(path, device="cpu"):
    return torch.load(path, map_location=device)

def plot_losses(train_losses, val_losses, outpath):
    plt.figure(figsize=(6,4))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=120)
    plt.close()
