import torch
from src.data import build_dataset, split_train_val, get_batch

def main():
    ds = build_dataset("data/train_essays_7_prompts.csv")
    train_ids, val_ids = split_train_val(ds.data_ids, val_ratio=0.1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, T = 4, 64
    x, y = get_batch(train_ids, batch_size=B, seq_len=T, device=device)

    print("vocab_size:", ds.vocab_size)
    print("train tokens:", train_ids.numel(), "val tokens:", val_ids.numel())
    print("x shape:", tuple(x.shape), "y shape:", tuple(y.shape), "device:", x.device)
    print("sample decode x[0][:80]:", ds.decode(x[0].tolist())[:80])

if __name__ == "__main__":
    main()