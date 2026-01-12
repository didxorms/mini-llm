from __future__ import annotations
import argparse
import time
import torch
import torch.nn.functional as F

from src.model import TinyLM


@torch.no_grad()
def bench(model: TinyLM, start_ids: torch.Tensor, gen_tokens: int, mode: str = "greedy", use_cache: bool = False):
    model.eval()
    idx = start_ids.clone()

    if idx.size(1) > model.max_len:
        idx = idx[:, -model.max_len :]
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.time()

    if use_cache:
        out = idx
        prefill_idx = out[:, -model.max_len :] if out.size(1) > model.max_len else out
        logits, past = model(prefill_idx, use_cache=True)

    for _ in range(gen_tokens):
        if not use_cache:
            logits = model(idx) 
            
        probs = F.softmax(logits[:, -1, :], dim=-1)

        if mode == "greedy":
            next_id = torch.argmax(probs, dim=-1, keepdim=True)
        else:
            next_id = torch.multinomial(probs, num_samples=1)

        idx = torch.cat([idx, next_id], dim=1)
        if idx.size(1) > model.max_len:
            idx = idx[:, -model.max_len :]

        if use_cache:
            logits, past = model(next_id, past_kvs=past, use_cache=True)
        
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
    p.add_argument("--cache", action="store_true")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)

    vocab_size = int(ckpt.get("vocab_size", 256))
    max_len = int(ckpt["max_len"])

    model = TinyLM(
        vocab_size=vocab_size,
        max_len=max_len,
        d_model=256,
        n_heads=8,
        n_layers=6,
        d_ff=1024,
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model"])

    ctx = min(args.context, max_len)
    start = torch.randint(0, vocab_size, (1, ctx), device=device)

    dt, tps = bench(model, start, args.gen_tokens, mode=args.mode, use_cache=args.cache)
    print(f"device={device} | context={ctx} | gen={args.gen_tokens} | {tps:.2f} tokens/s | {dt:.2f}s")


if __name__ == "__main__":
    main()