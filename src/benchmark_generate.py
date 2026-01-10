from __future__ import annotations
import argparse
import time
import torch
import torch.nn.functional as F

from src.model import TinyLM


@torch.no_grad()
def bench(model: TinyLM, start_ids: torch.Tensor, gen_tokens: int, mode: str = "greedy"):
    model.eval()
    idx = start_ids.clone()

    if idx.size(1) > model.max_len:
        idx = idx[:, -model.max_len :]
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(gen_tokens):
        logits = model(idx)
        probs = F.softmax(logits[:, -1, :], dim=-1)

        if mode == "greedy":
            next_id = torch.argmax(probs, dim=-1, keepdim=True)
        else:
            next_id = torch.multinomial(probs, num_samples=1)

        idx = torch.cat([idx, next_id], dim=1)
        if idx.size(1) > model.max_len:
            idx = idx[:, -model.max_len :]
        
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    dt = time.time() - t0
    tps = gen_tokens / dt
    return dt, tps

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/best.pt")
    p.add_argument("--context", type=int, default=128)
    p.add_argument("--gen_tokens", type=int, default=512)
    p.add_argument("--mode", choices=["sample", "greedy"], default="greedy")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)

    vocab = ckpt["vocab"]
    max_len = ckpt["max_len"]

    model = TinyLM(
        vocab_size=len(vocab),
        max_len=max_len,
        d_model=256,
        n_heads=8,
        n_layers=6,
        d_ff=1024,
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model"])

    ctx = min(args.context, max_len)
    start = torch.randint(0, len(vocab), (1, ctx), device=device)

    dt, tps = bench(model, start, args.gen_tokens, mode=args.mode)
    print(f"device={device} | context={ctx} | gen={args.gen_tokens} | {tps:.2f} tokens/s | {dt:.2f}s")


if __name__ == "__main__":
    main()