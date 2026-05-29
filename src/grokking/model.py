"""Minimal 2-layer transformer for grokking modular addition.

Mirrors Nanda et al. 2023 architecture: learned token+pos embedding, 2
transformer blocks (attn + MLP), unembed. No layernorm — Nanda's grokking
setup omits it (and including it makes the Fourier circuit harder to read).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)                       # [B, T, H, Dh]
        q = q.transpose(1, 2)                             # [B, H, T, Dh]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = scores.softmax(dim=-1)
        out = attn @ v                                    # [B, H, T, Dh]
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_mlp: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_mlp, bias=False)
        self.fc2 = nn.Linear(d_mlp, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_mlp: int):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.mlp = MLP(d_model, d_mlp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class GrokkingTransformer(nn.Module):
    """2-layer attn+MLP transformer. No LN, no biases (Nanda-style)."""

    def __init__(self, vocab: int, n_ctx: int = 3, d_model: int = 128,
                 n_heads: int = 4, d_mlp: int = 512, n_layers: int = 2):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Embedding(n_ctx, d_model)
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, d_mlp) for _ in range(n_layers)
        ])
        self.unembed = nn.Linear(d_model, vocab, bias=False)
        self.n_ctx = n_ctx

        # Nanda-style init: small weights to stay in the lazy regime briefly
        for p in self.parameters():
            if p.dim() >= 2:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        h = self.tok_emb(x) + self.pos_emb(pos)[None]
        for block in self.blocks:
            h = block(h)
        return self.unembed(h)            # [B, T, vocab]

    def logits_at_last(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)[:, -1, :]  # [B, vocab]
