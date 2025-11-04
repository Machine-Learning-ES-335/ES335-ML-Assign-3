#Load a checkpoint and visualize embeddings with t-SNE.

import argparse, torch, os, json
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    return ckpt

def main(args):
    device = "cpu"
    ckpt = load_checkpoint(args.model_path, device=device)
    vocab = ckpt["vocab"]
    args_saved = ckpt.get("args", {})
    # Reconstruct a tiny embedding array: load model state dict embeddings
    emb_key = None
    for k in ckpt["model_state"].keys():
        if "embedding" in k and "weight" in k:
            emb_key = k
            break
    if emb_key is None:
        raise RuntimeError("embedding key not found in checkpoint model_state")
    emb = ckpt["model_state"][emb_key].cpu().numpy()
    top_k = min(args.top_k, emb.shape[0])
    # select top_k most frequent words (vocab order matches training)
    selected_idx = list(range(top_k))
    emb_sel = emb[selected_idx]
    words = [vocab[i] for i in selected_idx]
    if emb_sel.shape[1] > 2:
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        proj = tsne.fit_transform(emb_sel)
    else:
        proj = emb_sel
    plt.figure(figsize=(10,10))
    plt.scatter(proj[:,0], proj[:,1], s=8)
    for i,w in enumerate(words):
        plt.annotate(w, (proj[i,0], proj[i,1]), fontsize=8)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print("Saved embedding plot to", args.out)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--dataset", choices=["warpeace","linux"], required=True)
    p.add_argument("--top_k", type=int, default=100)
    p.add_argument("--out", type=str, default="emb_viz.png")
    args = p.parse_args()
    main(args)
