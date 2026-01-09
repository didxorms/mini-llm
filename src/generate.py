from __future__ import annotations
import argparse
import torch
import torch.nn.functional as F

from src.data import build_dataset
from src.model import TinyLM

@torch.no_grad()
def generate(model, idx, max_new_tokens=200, temperature=1.0):
    model.eval()
    for _ in range(max_new_tokens):
        logits = model(idx)
        logits = logits[:, -1, :] / max(temperature, 1e-6)
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)

        if idx.size(1) > model.max_len:
            idx = idx[:, -model.max_len :]
    return idx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/tinylm_day2.pt")
    p.add_argument("--prompt", default="The ")
    p.add_argument("--tokens", type=int, default=300)
    p.add_argument("--temp", type=float, default=1.0)
    args = p.parse_args()

    ds = build_dataset("data/input.txt")
    
    ckpt = torch.load(args.ckpt, map_location="cpu")
    vocab = ckpt["vocab"]
    stoi = {ch: i for i, ch in enumerate(vocab)}

    ids = [stoi[ch] for ch in args.prompt if ch in stoi]
    if not ids:
        raise ValueError("Prompt has no known characters in vocab.")
    
    idx = torch.tensor([ids], dtype=torch.long)

    model = TinyLM(
        vocab_size=len(vocab),
        max_len=ckpt["max_len"],
        d_model=256,
        n_heads=8,
        n_layers=6,
        d_ff=1024,
        dropout=0.0,
    )
    model.load_state_dict(ckpt["model"])

    out = generate(model, idx, max_new_tokens=args.tokens, temperature=args.temp)
    text = "".join(vocab[i] for i in out[0].tolist())
    print(text)


if __name__ == "__main__":
    main()