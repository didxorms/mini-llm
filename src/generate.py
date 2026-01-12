from __future__ import annotations
import argparse
import random
import torch
import torch.nn.functional as F

from src.data import build_dataset
from src.model import TinyLM

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)

@torch.no_grad()
def generate(
    model: TinyLM,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    mode: str,
    top_k: int,
    use_cache: bool = False,
):
    model.eval()

    rep_penalty = 1.2
    window = 64
    
    if not use_cache:
        for _ in range(max_new_tokens):
            logits = model(idx)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            
            recent = idx[0, -window:].tolist()
            for tok in set(recent):
                logits[0, tok] /= rep_penalty

            probs = F.softmax(logits, dim=-1)

            if mode == "greedy":
                next_id = torch.argmax(probs, dim=-1, keepdim=True)
            else:
                if top_k > 0:
                    topk_probs, topk_idx = torch.topk(probs, k=top_k, dim=-1)
                    topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
                    next_local = torch.multinomial(topk_probs, num_samples=1)
                    next_id = torch.gather(topk_idx, dim=-1, index=next_local)
                else:
                    next_id = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, next_id], dim=1)

            if idx.size(1) > model.max_len:
                idx = idx[:, -model.max_len :]
        return idx
    
    out = idx

    prefill_idx = out[:, -model.max_len :] if out.size(1) > model.max_len else out

    logits, past = model(prefill_idx, use_cache=True)

    for _ in range(max_new_tokens):
        last = logits[:, -1, :] / max(temperature, 1e-6)

        recent = out[0, -window:].tolist()
        for tok in set(recent):
            last[0, tok] /= rep_penalty

        probs = F.softmax(last, dim=-1)

        if mode == "greedy":
            next_id = torch.argmax(probs, dim=-1, keepdim=True)
        else:
            if top_k > 0:
                topk_probs, topk_idx = torch.topk(probs, k=top_k, dim=-1)
                topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
                next_local = torch.multinomial(topk_probs, num_samples=1)
                next_id = torch.gather(topk_idx, dim=-1, index=next_local)
            else:
                next_id = torch.multinomial(probs, num_samples=1)

        out = torch.cat([out, next_id], dim=1)

        logits, past = model(next_id, past_kvs=past, use_cache=True)
    
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/tinylm_day2.pt")
    p.add_argument("--prompt", default="After people saw ")
    p.add_argument("--tokens", type=int, default=300)
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--mode", choices=["sample", "greedy"], default="sample")
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cache", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = build_dataset("data/input.txt")
    ckpt = torch.load(args.ckpt, map_location="cpu")

    prompt_bytes = args.prompt.encode("utf-8")
    ids = list(prompt_bytes)
    if len(ids) == 0:
        raise ValueError("Empty prompt.")
    idx = torch.tensor([ids], dtype=torch.long, device=device)

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

    out = generate(
        model=model,
        idx=idx,
        max_new_tokens=args.tokens,
        temperature=args.temp,
        mode=args.mode,
        top_k=args.top_k,
        use_cache=args.cache,
    )

    out_ids = out[0].tolist()
    print(ds.decode(out_ids))


if __name__ == "__main__":
    main()