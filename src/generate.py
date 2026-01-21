from __future__ import annotations
import argparse
import random
import torch
import torch.nn.functional as F

from src.data import build_dataset, build_spm_dataset
from src.model import TinyLM

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)

def sample_next_id(
    logits: torch.Tensor,
    mode: str,
    top_k: int = 0,
    top_p: float = 0.0,
) -> torch.Tensor:
    if mode == "greedy":
        return torch.argmax(logits, dim=-1, keepdim=True)

    probs = F.softmax(logits, dim=-1)

    if top_k and top_k > 0:
        topk_probs, topk_idx = torch.topk(probs, k=top_k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        next_local = torch.multinomial(topk_probs, num_samples=1)
        return torch.gather(topk_idx, dim=-1, index=next_local)
    
    if top_p and top_p > 0.0 and top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cum = torch.cumsum(sorted_probs, dim=-1)

        keep = cum <= top_p
        keep[..., 0] = True

        filtered = sorted_probs * keep
        filtered = filtered / filtered.sum(dim=-1, keepdim=True)

        next_local = torch.multinomial(filtered, num_samples=1)
        return torch.gather(sorted_idx, dim=-1, index=next_local)
    
    return torch.multinomial(probs, num_samples=1)

def apply_no_repeat_ngram(
    logits: torch.Tensor,
    idx: torch.Tensor,
    n: int,
):
    if n <= 0:
        return logits
    B, T = idx.shape
    if T < n:
        return logits
    
    for b in range(B):
        seq = idx[b].tolist()
        prefix = tuple(seq[-(n-1):]) if n > 1 else tuple()

        banned = set()
        for i in range(T - n + 1):
            gram = tuple(seq[i:i+n])
            if n == 1:
                banned.add(gram[0])
            else:
                if gram[:-1] == prefix:
                    banned.add(gram[-1])
        
        if banned:
            logits[b, list(banned)] = float("-inf")
    return logits

@torch.no_grad()
def generate(
    model: TinyLM,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    mode: str,
    top_k: int,
    top_p: float = 0.0,
    no_repeat_ngram: int = 0,
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

            logits = apply_no_repeat_ngram(logits, idx, no_repeat_ngram)
            next_id = sample_next_id(logits, mode, top_k, top_p)

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

        last = apply_no_repeat_ngram(last, out, no_repeat_ngram)
        next_id = sample_next_id(last, mode, top_k, top_p)

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
    p.add_argument("--top_p", type=float, default=0.0, help="nucleus sampling (0 disables)")
    p.add_argument("--no_repeat_ngram", type=int, default=0, help="0 disables; e.g. 3 blocks repeating 3-grams")
    p.add_argument("--tokenizer", choices=["byte", "spm"], default=None)
    p.add_argument("--spm_model", default=None)
    args = p.parse_args()

    set_seed(args.seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = build_dataset("data/input.txt")
    ckpt = torch.load(args.ckpt, map_location="cpu")

    tok = args.tokenizer or ckpt.get("tokenizer", "byte")
    spm_model = args.spm_model or ckpt.get("spm_model", "data/spm_unigram_4k.model")

    data_path = "data/train_essays_RDizzl3_seven_v2.csv"
    if tok == "spm":
        ds = build_spm_dataset(data_path, spm_model)
        ids = ds.encode(args.prompt)
    else:
        ds = build_dataset(data_path)
        ids = list(args.prompt.encode("utf-8"))

    if len(ids) == 0:
        raise ValueError("Empty prompt.")
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    vocab_size = int(ckpt["vocab_size"])
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
        top_p=args.top_p,
        no_repeat_ngram=args.no_repeat_ngram,
        use_cache=args.cache,
    )

    out_ids = out[0].tolist()
    print(ds.decode(out_ids))


if __name__ == "__main__":
    main()
