"""Compact GPT-style causal language model with optional checkpointing."""

import math, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout, block_size, n_kv_head: int | None = None, use_sdpa: bool = False):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_kv_head = n_kv_head if (n_kv_head is not None and n_kv_head > 0 and n_kv_head <= n_head) else None
        self.use_sdpa = bool(use_sdpa)
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).unsqueeze(0).unsqueeze(0))

    def forward(self, x, attn_mask=None):
        B, T, C = x.size()
        head_dim = C // self.n_head
        q = self.query(x).view(B, T, self.n_head, head_dim).transpose(1,2)
        if self.n_kv_head is None:
            k = self.key(x).view(B, T, self.n_head, head_dim).transpose(1,2)
            v = self.value(x).view(B, T, self.n_head, head_dim).transpose(1,2)
        else:
            # Grouped-query attention: fewer KV heads broadcast to query heads
            if self.n_head % self.n_kv_head != 0:
                raise ValueError("n_head must be divisible by n_kv_head for GQA")
            kv_dim = C // self.n_kv_head
            kv_head_dim = kv_dim // self.n_kv_head  # equals head_dim when dims align
            k = self.key(x).view(B, T, self.n_kv_head, head_dim * (self.n_head // self.n_kv_head)).transpose(1,2)
            v = self.value(x).view(B, T, self.n_kv_head, head_dim * (self.n_head // self.n_kv_head)).transpose(1,2)
            # Repeat along heads to match n_head
            rep = self.n_head // self.n_kv_head
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        base = self.mask[:,:,:T,:T]
        if self.use_sdpa and hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            # Build boolean mask combine causal with optional segment mask
            if attn_mask is not None:
                causal = (base > 0)
                segmask = (attn_mask > 0)
                # both are (B,1,T,T); broadcast to heads
                mask = ~(causal & segmask).expand(B, self.n_head, T, T)
            else:
                mask = ~(base > 0).expand(B, self.n_head, T, T)
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
            y = y.transpose(1,2).contiguous().view(B, T, C)
        else:
            att = (q @ k.transpose(-2,-1)) / math.sqrt(k.size(-1))
            if attn_mask is not None:
                if attn_mask.dtype != base.dtype:
                    attn_mask = attn_mask.to(base.dtype)
                combined = (base > 0) & (attn_mask > 0)
                att = att.masked_fill(~combined, float('-inf'))
            else:
                att = att.masked_fill(base==0, float('-inf'))
            att = torch.softmax(att, dim=-1)
            y = att @ v
            y = y.transpose(1,2).contiguous().view(B, T, C)
        return self.proj(y)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.GELU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.ln1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_layer=3, n_head=4, n_embd=256, dropout=0.1, use_checkpoint=False, label_smoothing: float = 0.0, sep_id: int | None = 3, tie_embeddings: bool = True, n_kv_head: int | None = None, use_sdpa: bool = False):
        super().__init__()
        self.block_size = block_size
        self.use_checkpoint = use_checkpoint
        self.label_smoothing = float(label_smoothing)
        self.sep_id = sep_id
        self.tie_embeddings = bool(tie_embeddings)
        self.n_kv_head = n_kv_head if (n_kv_head is not None and n_kv_head > 0) else None
        self.use_sdpa = bool(use_sdpa)
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, dropout, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        if self.tie_embeddings:
            self.head.weight = self.tok_emb.weight

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)

        attn_mask = None
        if self.sep_id is not None:
            # Build block-diagonal segment mask from <SEP> boundaries
            sep = (idx == int(self.sep_id))
            # cumulative segment id increments after SEP; ensure SEP belongs to previous segment
            seg = torch.cumsum(sep, dim=1)
            # allow attention only within equal seg ids
            attn_mask = (seg.unsqueeze(-1) == seg.unsqueeze(-2))  # (B,T,T)
            attn_mask = attn_mask.unsqueeze(1)  # (B,1,T,T)

        if self.use_checkpoint and self.training:
            for blk in self.blocks:
                # Explicitly pass use_reentrant to silence future deprecation warnings.
                # Fallback for older PyTorch that doesn't support the kwarg.
                try:
                    x = checkpoint(lambda _x: blk(_x, attn_mask=attn_mask), x, use_reentrant=False)
                except TypeError:
                    x = checkpoint(lambda _x: blk(_x, attn_mask=attn_mask), x)
        else:
            for blk in self.blocks:
                x = blk(x, attn_mask=attn_mask)

        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            # Compute loss in float32 to avoid potential NaNs under AMP on some backends
            loss = F.cross_entropy(
                logits.float().view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=0,
                label_smoothing=self.label_smoothing,
            )
        return logits, loss


__all__ = ["TinyGPT"]
