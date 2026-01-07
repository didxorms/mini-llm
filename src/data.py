from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import torch

@dataclass
class TextDataset:
    text: str
    stoi: dict[str, int]
    itos: list[str]
    data_ids: torch.Tensor

    @property
    def vocab_size(self) -> int:
        return len(self.itos)
    
    def encode(self, s: str) -> list[int]:
        return [self.stoi[ch] for ch in s]
    
    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)
    
def build_dataset(txt_path: str | Path) -> TextDataset:
    path = Path(txt_path)
    text = path.read_text(encoding='utf-8')

    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = chars
    
    ids = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    return TextDataset(text=text, stoi=stoi, itos=itos, data_ids=ids)

def split_train_val(ids: torch.Tensor, val_ratio: float = 0.1):
    n = ids.numel()
    n_val = int(n * val_ratio)
    train_ids = ids[:-n_val] if n_val > 0 else ids
    val_ids = ids[-n_val:] if n_val > 0 else ids[:0]
    return train_ids, val_ids

def get_batch(
        ids: torch.Tensor,
        batch_size: int,
        seq_len: int,
        device: str = "cpu",
):
    if ids.numel() < seq_len + 1:
        raise ValueError(f"Data too small: need >= {seq_len+1} tokens, got {ids.numel()}")
    
    max_start = ids.numel() - (seq_len + 1)
    starts = torch.randint(0, max_start + 1, (batch_size,))
    x = torch.stack([ids[s : s + seq_len] for s in starts], dim=0)
    y = torch.stack([ids[s + 1 : s + 1 + seq_len] for s in starts], dim=0)
    return x.to(device), y.to(device)
