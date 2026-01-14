from __future__ import annotations
import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from src.data import build_dataset, build_spm_dataset, split_train_val, get_batch
from src.model import TinyLM

def save_ckpt(path: str, model: TinyLM, optim: torch.optim.Optimizer, vocab_size: int, step:int, max_len: int, args):
    ckpt = {
        "step": step,
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "tokenizer": args.tokenizer,
        "spm_model": args.spm_model if args.tokenizer == "spm" else None,
        "vocab_size": vocab_size,
        "max_len": max_len,
        "args": vars(args),
    }
    torch.save(ckpt, path)


@torch.no_grad()
def estimate_loss(model, train_ids, val_ids, device, batch_size, seq_len, iters=30):
    model.eval()
    out = {}
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
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/train_essays_RDizzl3_seven_v2.csv")
    p.add_argument("--tokenizer", choices=["byte", "spm"], default="byte")
    p.add_argument("--spm_model", default="data/spm_unigram_4k.model")
    p.add_argument("--max_steps", type=int, default=5000)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=0, help="0이면 디바이스에 따라 자동 설정")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--resume", default="", help="checkpoint 경로 (없으면 새로 학습)")
    p.add_argument("--out_dir", default="runs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_file", default="", help="예: runs/train_log.txt (비우면 파일 저장 안 함)")
    args = p.parse_args()

    log_f = None
    if args.log_file:
        Path(args.log_file).parent.mkdir(exist_ok=True, parents=True)
        log_f = open(args.log_file, "a", encoding="utf-8")

    def log(msg: str):
        print(msg)
        if log_f:
            log_f.write(msg + "\n")
            log_f.flush()

    torch.manual_seed(args.seed)

    if args.tokenizer == "spm":
        ds = build_spm_dataset(args.data, args.spm_model)
    else:
        ds = build_dataset(args.data)
    train_ids, val_ids = split_train_val(ds.data_ids, val_ratio=0.1)

    device = "cuda" if torch.cuda.is_available() else "cpu"    
    batch_size = args.batch_size
    if batch_size == 0:
        batch_size = 16 if device == "cuda" else 8

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    model_kwargs = dict(
        vocab_size=ds.vocab_size,
        max_len=args.seq_len,
        d_model=256,
        n_heads=8,
        n_layers=6,
        d_ff=1024,
        dropout=args.dropout,
    )

    model = TinyLM(**model_kwargs).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_step = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])
        start_step = int(ckpt.get("step", 0))
        log(f"[resume] loaded {args.resume} (start_step={start_step})")

    log(f"device: {device} | vocab: {ds.vocab_size} | seq_len: {args.seq_len} | batch: {batch_size}")
    print(f"max_steps: {args.max_steps} | eval_every: {args.eval_every} | save_every: {args.save_every}")

    t0 = time.time()
    
    best_val = float("inf")
    for step in range(start_step + 1, args.max_steps + 1):
        x, y = get_batch(train_ids, batch_size=batch_size, seq_len=args.seq_len, device=device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        if step % 50 == 0:
            dt = time.time() - t0
            log(f"step {step:6d} | loss {loss.item():.4f} | ppl {math.exp(loss.item()):.2f} | {dt:.1f}s")

        if step % args.eval_every == 0:
            est = estimate_loss(model, train_ids, val_ids, device, batch_size, args.seq_len, iters=30)
            print(f"[eval] train {est['train']:.4f} | val {est['val']:.4f}")

            if est["val"] < best_val:
                best_val = est["val"]
                best_path = out_dir / f"best_{args.tokenizer}.pt"
                save_ckpt(
                    str(best_path),
                    model, optim, ds.vocab_size, step,
                    max_len=args.seq_len,
                    args = args,
                )
                print(f"[save] best -> {best_path} (val={best_val:.4f})")
        
        if step % args.save_every == 0:
            ckpt_path = out_dir / f"ckpt_step_{step}.pt"
            save_ckpt(
                str(ckpt_path),
                model, optim, ds.vocab_size, step,
                max_len=args.seq_len,
                args = args,
            )
            log(f"[save] {ckpt_path}")

    final_path = out_dir / f"final_{args.tokenizer}.pt"
    save_ckpt(
        str(final_path),
        model, optim, ds.vocab_size, step,
        max_len=args.seq_len,
        args = args,
    )
    print(f"[save] final -> {final_path}")

    if log_f:
        log_f.close()


if __name__ == "__main__":
    main()