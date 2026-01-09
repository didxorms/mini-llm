from __future__ import annotations
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from src.data import build_dataset, split_train_val, get_batch
from src.model import TinyLM

def estimate_loss(model, train_ids, val_ids, device, batch_size, seq_len, iters=50):
    model.eval()
    out = {}
    with torch.no_grad():
        for split, ids in [("train", train_ids), ("val", val_ids)]:
            losses = []
            for _ in range(iters):
                x, y = get_batch(ids, batch_size=batch_size, seq_len=seq_len, device=device)
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                losses.append(loss.item())
            out[split] = sum(losses) / len(losses)
    model.train()
    return out

def main():
    torch.manual_seed(42)

    ds = build_dataset("data/train_essays_7_prompts.csv")
    train_ids, val_ids = split_train_val(ds.data_ids, val_ratio=0.1)

    device = "cuda" if torch.cuda.is_available() else "cpu"    
    print("device:", device, "| vocab:", ds.vocab_size)

    seq_len = 128
    batch_size = 16 if device == "cuda" else 8

    model = TinyLM(
        vocab_size = ds.vocab_size,
        max_len=seq_len,
        d_model=256,
        n_heads=8,
        n_layers=6,
        d_ff=1024,
        dropout=0.1,
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

    max_stemps = 500
    log_every = 50
    eval_every = 200
    t0 = time.time()

    for step in range(1, max_stemps + 1):
        x, y = get_batch(train_ids, batch_size=batch_size, seq_len=seq_len, device=device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        if step % log_every == 0:
            dt = time.time() - t0
            print(f"step {step:4d} | loss {loss.item():.4f} | ppl {math.exp(loss.item()):.2f} | {dt:.1f}s")

        if step % eval_every == 0:
            est = estimate_loss(model, train_ids, val_ids, device, batch_size, seq_len, iters=30)
            print(f"[eval] train {est['train']:.4f} | val {est['val']:.4f}")

    Path("runs").mkdir(exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
        "vocab": ds.itos,
        "max_len": seq_len,
    }
    torch.save(ckpt, "runs/tinylm_day2.pt")
    print("saved:", "runs/tinylm_day2.pt")

if __name__ == "__main__":
    main()