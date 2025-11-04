#!/usr/bin/env python3
#Train a word-level MLP next-word predictor and save checkpoints.

import argparse, os, json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import NextWordDataset as NextWordDataset

from utils import save_checkpoint, plot_losses
from tqdm import tqdm

class MLPNextWord(nn.Module):
    def __init__(self, vocab_size, context_len=5, embed_dim=64, hidden_size=512, n_hidden=1, activation="relu", dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        layers = []
        in_dim = context_len * embed_dim
        for _ in range(n_hidden):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU() if activation=="relu" else nn.Tanh())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_size
        layers.append(nn.Linear(in_dim, vocab_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, context_len)
        e = self.embedding(x)  # (B, context_len, embed_dim)
        flat = e.view(e.size(0), -1)
        return self.net(flat)

def evaluate(model, loader, device, criterion):
    model.eval()
    tot_loss = 0.0
    tot = 0
    correct = 0
    with torch.no_grad():
        for X,y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            tot_loss += loss.item() * X.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds==y).sum().item()
            tot += X.size(0)
    return tot_loss/tot, correct/tot

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    print("Device:", device)
    train_ds = NextWordDataset(args.dataset, context_len=args.context_len, split="train", split_ratio=1-args.val_ratio)
    val_ds = NextWordDataset(args.dataset, context_len=args.context_len, split="val", split_ratio=1-args.val_ratio)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    vocab_size = len(train_ds.vocab)
    model = MLPNextWord(vocab_size, context_len=args.context_len, embed_dim=args.embed_dim, hidden_size=args.hidden_size, n_hidden=args.n_hidden, activation=args.activation, dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.save_dir, exist_ok=True)
    best_val = float("inf")
    train_losses, val_losses = [], []
    for epoch in range(1, args.epochs+1):
        model.train()
        epoch_loss = 0.0
        tot = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for X,y in pbar:
            X,y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X.size(0)
            tot += X.size(0)
            pbar.set_postfix({"loss": epoch_loss/tot})
        train_loss = epoch_loss/tot
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        print(f"Epoch {epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        train_losses.append(train_loss); val_losses.append(val_loss)
        # save best
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "vocab": train_ds.vocab,
                "args": vars(args)
            }, os.path.join(args.save_dir, "best.pt"))
        if epoch % args.save_every == 0:
            save_checkpoint({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "vocab": train_ds.vocab,
                "args": vars(args)
            }, os.path.join(args.save_dir, f"epoch_{epoch}.pt"))
    # final save
    save_checkpoint({
        "epoch": args.epochs,
        "model_state": model.state_dict(),
        "vocab": train_ds.vocab,
        "args": vars(args)
    }, os.path.join(args.save_dir, "final.pt"))
    plot_losses(train_losses, val_losses, os.path.join(args.save_dir, "losses.png"))
    print("Training complete. Models in", args.save_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["warpeace","linux"], required=True)
    parser.add_argument("--context_len", type=int, default=5)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--n_hidden", type=int, default=1)
    parser.add_argument("--activation", choices=["relu","tanh"], default="relu")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_dir", type=str, default="models")
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--force_cpu", action="store_true")
    args = parser.parse_args()
    # create save_dir per dataset to avoid collisions
    args.save_dir = os.path.join(args.save_dir, args.dataset)
    os.makedirs(args.save_dir, exist_ok=True)
    main(args)
