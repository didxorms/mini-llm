from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import torch
import sentencepiece as spm

@dataclass
class ByteDataset:
    raw_bytes: bytes
    data_ids: torch.Tensor

    @property
    def vocab_size(self) -> int:
        return 256
    
    def decode(self, ids: torch.Tensor | list[int]) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        b = bytes([int(x) & 0xFF for x in ids])
        return b.decode("utf-8", errors="replace")

@dataclass
class SPMDataset:
    sp: spm.SentencePieceProcessor
    data_ids: torch.Tensor

    @property
    def vocab_size(self) -> int:
        return int(self.sp.get_piece_size())
    
    def encode(self, text: str) -> list[int]:
        return list(self.sp.encode(text, out_type=int))
    
    def decode(self, ids: torch.Tensor | list[int]) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self.sp.decode([int(x) for x in ids])

def build_dataset(txt_path: str | Path) -> ByteDataset:
    txt_path = Path(txt_path)
    raw = txt_path.read_bytes()
    data_ids = torch.tensor(list(raw), dtype=torch.long)
    return ByteDataset(raw_bytes=raw, data_ids=data_ids)

def build_spm_dataset(txt_path: str | Path, spm_model_path: str | Path) -> SPMDataset:
    txt_path = Path(txt_path)
    spm_model_path = Path(spm_model_path)

    sp = spm.SentencePieceProcessor()
    sp.load(str(spm_model_path))

    text = txt_path.read_text(encoding="utf-8", errors="replace")
    ids = sp.encode(text, out_type=int)
    data_ids = torch.tensor(ids, dtype=torch.long)
    return SPMDataset(sp=sp, data_ids=data_ids)

def split_train_val(data_ids: torch.Tensor, val_ratio: float = 0.1):
    n = data_ids.numel()
    n_val = int(n * val_ratio)
    train_ids = data_ids[:-n_val]
    val_ids = data_ids[-n_val:]
    return train_ids, val_ids

def get_batch(data_ids: torch.Tensor, batch_size: int, seq_len: int, device: str):
    N = data_ids.numel()
    ix = torch.randint(0, N - seq_len -1, (batch_size,))
    x = torch.stack([data_ids[i : i + seq_len] for i in ix])
    y = torch.stack([data_ids[i + 1 : i + 1 + seq_len] for i in ix])
    return x.to(device), y.to(device)
