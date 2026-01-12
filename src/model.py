from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, past_kv=None):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=-1)

        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        past_len = 0
        if past_kv is not None:
            pk, pv = past_kv
            past_len = pk.size(2)
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        
        Tk = k.size(2)
        att = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)

        mask = torch.tril(torch.ones(T, Tk, device=x.device), diagonal=past_len).view(1, 1, T, Tk)
        att = att.masked_fill(mask == 0, float("-inf"))

        w = F.softmax(att, dim=-1)
        w = self.drop(w)
        y = w @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        return y, (k, v)
    
class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1= nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, past_kv=None, use_cache=False):
        a, new_kv = self.attn(self.ln1(x), past_kv=past_kv)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        if use_cache:
            return x, new_kv
        return x
    
class TinyLM(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            max_len: int = 256,
            d_model: int = 256,
            n_heads: int = 8,
            n_layers: int = 6,
            d_ff: int = 1024,
            dropout: float = 0.1,
    ):
        super().__init__()
        self.max_len = max_len
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx, past_kvs=None, use_cache: bool = False):
        B, T = idx.shape
        device = idx.device

        past_len = 0
        if past_kvs is not None:
            past_len = past_kvs[0][0].size(2)

            if past_len + T > self.max_len:
                keep = self.max_len - T
                past_kvs = [(k[:, :, -keep:, :].contiguous(), v[:, :, -keep:, :].contiguous()) for (k, v) in past_kvs]
                past_len = keep

        pos = torch.arange(past_len, past_len + T, device=device).unsqueeze(0)
        x = self.tok(idx) + self.pos(pos)
        x = self.drop(x)

        new_past = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            pkv = past_kvs[i] if past_kvs is not None else None
            if use_cache:
                x, kv = block(x, past_kv=pkv, use_cache=True)
                new_past.append(kv)
            else:
                x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)

        if use_cache:
            return logits, new_past
        return logits