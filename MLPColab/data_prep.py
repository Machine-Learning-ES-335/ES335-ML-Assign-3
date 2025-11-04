#!/usr/bin/env python3
# Download and preprocess the two datasets:
# - warpeace (Tolstoy)
# - linux (linux_input)
# Saves:
#   data/<dataset>/vocab.json
#   data/<dataset>/tokens.ids

import argparse, os, re, urllib.request, json
from collections import Counter

DATA_DIR = "data"
URLS = {
    "warpeace": "https://cs.stanford.edu/people/karpathy/char-rnn/warpeace_input.txt",
    "linux": "https://cs.stanford.edu/people/karpathy/char-rnn/linux_input.txt"
}

SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]

def download(url, outpath):
    if os.path.exists(outpath):
        print("Already downloaded:", outpath)
        return
    print("Downloading", url)
    urllib.request.urlretrieve(url, outpath)
    print("Saved to", outpath)

def preprocess_text(path, keep_dot=True):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().lower()
    if keep_dot:
        text = re.sub(r"[^a-z0-9 \.\n]", " ", text)
    else:
        text = re.sub(r"[^a-z0-9 \n]", " ", text)
    # collapse spaces
    text = re.sub(r"[ ]+", " ", text)
    # convert multiple newlines to dot sentinel to mark paragraph breaks
    text = re.sub(r"\n+", " . ", text)
    tokens = text.split()
    return tokens

def build_vocab(tokens, min_freq=1):
    c = Counter(tokens)
    vocab = [t for t,f in c.items() if f >= min_freq]
    # Sort by frequency desc
    vocab = sorted(vocab, key=lambda w: (-c[w], w))
    final = SPECIAL_TOKENS + [w for w in vocab if w not in SPECIAL_TOKENS]
    freqs = {w:c[w] for w in final}
    w2i = {w:i for i,w in enumerate(final)}
    i2w = {i:w for w,i in w2i.items()}
    return final, w2i, i2w, freqs

def save_vocab(outdir, vocab, freqs):
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump({"vocab":vocab, "freqs":freqs}, f, ensure_ascii=False, indent=2)

def save_ids(outdir, ids):
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "tokens.ids"), "w", encoding="utf-8") as f:
        f.write(" ".join(map(str, ids)))

def tokens_to_ids(tokens, w2i):
    unk = w2i.get("<UNK>")
    return [w2i.get(t, unk) for t in tokens]

def main(args):
    ds = args.dataset
    assert ds in URLS, "dataset must be warpeace or linux"
    os.makedirs(DATA_DIR, exist_ok=True)
    raw = os.path.join(DATA_DIR, f"{ds}_raw.txt")
    download(URLS[ds], raw)
    print("Preprocessing...")
    tokens = preprocess_text(raw, keep_dot=(ds=="warpeace"))
    print("Total tokens:", len(tokens))
    vocab, w2i, i2w, freqs = build_vocab(tokens, min_freq=args.min_freq)
    print("Vocab size:", len(vocab))
    print("Top 10 frequent:", list(sorted(freqs.items(), key=lambda x:-x[1]))[:10])
    print("Least 10 frequent:", list(sorted(freqs.items(), key=lambda x:x[1]))[:10])
    outdir = os.path.join(DATA_DIR, ds)
    save_vocab(outdir, vocab, freqs)
    ids = tokens_to_ids(tokens, w2i)
    save_ids(outdir, ids)
    print("Saved to", outdir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["warpeace","linux"], required=True)
    parser.add_argument("--min_freq", type=int, default=1)
    args = parser.parse_args()
    main(args)
