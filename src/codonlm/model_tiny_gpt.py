"""
Minimal GPT-like causal LM for codon tokens.

Design:
- Embedding → [Block]*n → LayerNorm → Linear(vocab)
- Each Block: (LN → MH-Attn → residual) + (LN → MLP → residual)
- Causal mask ensures token t can only attend to < t

Why this instead of HuggingFace?
- 120 lines, easy to read/modify.
- No heavy tokenizer dependency.

Notes for Apple Silicon:
- Keep dtype float32 (MPS is reliable there).
"""

import math, torch
import torch.nn as nn
import torch.nn.functional as F

class Cfg:
    def __init__(self, vocab_size, n_layer, n_head, n_embd, block_size, dropout=0.0):
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.block_size = block_size
        self.dropout = dropout

    def __getitem__(self, k): return getattr(self, k)



class CausalSelfAttention(nn.Module):
    def __init__(self, config: Cfg):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.block_size = config.block_size   # <- ensure this exists

        head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0

        self.qkv  = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)
        self.proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

        # register a causal mask up to block_size
        mask = torch.tril(torch.ones(self.block_size, self.block_size))
        self.register_buffer("attn_mask", mask.view(1, 1, self.block_size, self.block_size))

    def forward(self, x):
        B, T, C = x.size()
        if T > self.block_size:
            raise ValueError(f"Sequence length {T} exceeds block_size {self.block_size}")

        qkv = self.qkv(x)                               # (B,T,3C)
        q, k, v = qkv.chunk(3, dim=-1)                  # (B,T,C) each
        # reshape to heads
        B, T, C = q.size()
        H = self.n_head
        q = q.view(B, T, H, C // H).transpose(1, 2)     # (B,H,T,hd)
        k = k.view(B, T, H, C // H).transpose(1, 2)
        v = v.view(B, T, H, C // H).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / (q.size(-1) ** 0.5)    # (B,H,T,T)
        att_mask = self.attn_mask[:, :, :T, :T]
        att = att.masked_fill(att_mask == 0, float("-inf"))
        att = att.softmax(dim=-1)
        att = self.attn_drop(att)

        y = att @ v                                             # (B,H,T,hd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)        # (B,T,C)
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    def __init__(self, config: Cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)    # <- pass config through
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, config: Cfg):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size,  config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        if T > self.config.block_size:
            raise ValueError(f"Sequence length {T} exceeds block_size {self.config.block_size}")
        pos = torch.arange(0, T, device=idx.device)

        x = self.tok_emb(idx) + self.pos_emb(pos)              # (B,T,C)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

