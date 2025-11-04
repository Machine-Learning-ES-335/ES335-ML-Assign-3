# This app loads saved checkpoints from models/<dataset>/best.pt (or final.pt)
# and lets you generate next-k words using temperature sampling.

import streamlit as st
import torch, json, os
import numpy as np

st.set_page_config(page_title="Next-Word MLP (War & Linux)", layout="wide")

@st.cache_data
def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu")
    return ckpt

def build_model_from_ckpt(ckpt):
    # Lightweight model construction to extract embeddings and run forward
    args = ckpt.get("args", {})
    dataset_args = args
    vocab = ckpt["vocab"]
    vocab_size = len(vocab)
    context_len = dataset_args.get("context_len", 5)
    embed_dim = dataset_args.get("embed_dim", 64)
    hidden_size = dataset_args.get("hidden_size", 512)
    n_hidden = dataset_args.get("n_hidden", 1)
    activation = dataset_args.get("activation", "relu")
    # create model with same state dict shape: a minimal class to hold embedding and net
    import torch.nn as nn
    class TinyNet(nn.Module):
        def __init__(self, vocab_size, context_len, embed_dim, hidden_size, n_hidden, activation):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            layers = []
            in_dim = context_len * embed_dim
            for _ in range(n_hidden):
                layers.append(nn.Linear(in_dim, hidden_size))
                layers.append(nn.ReLU() if activation=="relu" else nn.Tanh())
                in_dim = hidden_size
            layers.append(nn.Linear(in_dim, vocab_size))
            self.net = nn.Sequential(*layers)
        def forward(self, x):
            e = self.emb(x)
            return self.net(e.view(e.size(0), -1))
    m = TinyNet(vocab_size, context_len, embed_dim, hidden_size, n_hidden, activation)
    m.load_state_dict(ckpt["model_state"], strict=False)
    m.eval()
    return m, vocab

def sample_from_logits(logits, temperature=1.0):
    probs = torch.softmax(logits/temperature, dim=-1).cpu().numpy().ravel()
    if temperature <= 0.0:
        return int(np.argmax(probs))
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))

def generate(model, vocab, seed_tokens, k=20, context_len=5, temperature=1.0):
    word2idx = {w:i for i,w in enumerate(vocab)}
    idx2word = {i:w for i,w in enumerate(vocab)}
    tokens = [word2idx.get(w, word2idx.get("<UNK>")) for w in seed_tokens]
    out = seed_tokens.copy()
    for _ in range(k):
        context = tokens[-context_len:] if len(tokens)>=context_len else [word2idx["<PAD>"]]*(context_len-len(tokens))+tokens
        x = torch.tensor([context], dtype=torch.long)
        logits = model(x)[0]
        nid = sample_from_logits(logits, temperature)
        tokens.append(nid)
        out.append(idx2word.get(nid, "<UNK>"))
    return " ".join(out)

st.title("Next-Word MLP Generator — War & Linux")

st.sidebar.header("Model & Generation")
models_dir = st.sidebar.text_input("Models root folder", value="models")
dataset = st.sidebar.selectbox("Dataset", ["warpeace","linux"])
# pick model file
model_path_default = os.path.join(models_dir, dataset, "best.pt")
model_path = st.sidebar.text_input("Model checkpoint path", value=model_path_default)
k = st.sidebar.number_input("Words to generate", min_value=1, max_value=200, value=30)
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=2.0, value=1.0)
context_len = st.sidebar.number_input("Context length", min_value=1, max_value=20, value=5)

st.sidebar.markdown("---")
if not os.path.exists(model_path):
    st.sidebar.error(f"Model not found: {model_path}")
else:
    ckpt = load_checkpoint(model_path)
    model, vocab = build_model_from_ckpt(ckpt)
    st.sidebar.success("Model loaded")
    st.write("Vocab size:", len(vocab))

seed_text = st.text_area("Seed text (words):", value="the")
btn = st.button("Generate")
if btn and os.path.exists(model_path):
    seed_tokens = seed_text.strip().lower().split()
    with st.spinner("Generating..."):
        out = generate(model, vocab, seed_tokens, k=k, context_len=context_len, temperature=temperature)
    st.subheader("Generated")
    st.write(out)
