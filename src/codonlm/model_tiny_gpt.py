"""Compact GPT-style causal language model with optional checkpointing."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout, block_size, n_kv_head: int | None = None, use_sdpa: bool = False):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_kv_head = n_kv_head if (n_kv_head is not None and n_kv_head > 0 and n_kv_head <= n_head) else None
        self.use_sdpa = bool(use_sdpa)

        head_dim = n_embd // n_head
        kv_dim = (self.n_kv_head * head_dim) if self.n_kv_head is not None else n_embd

        self.key = nn.Linear(n_embd, kv_dim)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, kv_dim)
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
            k = self.key(x).view(B, T, self.n_kv_head, head_dim).transpose(1,2)
            v = self.value(x).view(B, T, self.n_kv_head, head_dim).transpose(1,2)
            # Repeat along heads to match n_head
            rep = self.n_head // self.n_kv_head
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        base = self.mask[:,:,:T,:T]
        if self.use_sdpa and hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            if attn_mask is not None:
                # Ensure boolean mask for inversion/expansion
                attn_mask_bool = (attn_mask > 0)
                mask = attn_mask_bool.expand(B, self.n_head, T, T)
                y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
            else:
                # Trigger the highly optimized fused causal kernel
                y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
            y = y.transpose(1,2).contiguous().view(B, T, C)
        else:
            att = (q @ k.transpose(-2,-1)) / math.sqrt(k.size(-1))
            if attn_mask is not None:
                attn_mask_bool = (attn_mask > 0)
                att = att.masked_fill(~attn_mask_bool, float('-inf'))
            else:
                att = att.masked_fill(base==0, float('-inf'))
            att = torch.softmax(att, dim=-1)
            self.last_attn = att.detach()
            y = att @ v
            y = y.transpose(1,2).contiguous().view(B, T, C)
        return self.proj(y)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout, block_size, n_kv_head: int | None = None, use_sdpa: bool = False):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout, block_size, n_kv_head=n_kv_head, use_sdpa=use_sdpa)
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
    def __init__(
        self,
        vocab_size,
        block_size,
        n_layer=3,
        n_head=4,
        n_embd=256,
        dropout=0.1,
        use_checkpoint=False,
        label_smoothing: float = 0.0,
        sep_id: int | None = 3,
        tie_embeddings: bool = True,
        n_kv_head: int | None = None,
        use_sdpa: bool = False,
        loss_weights: list[float] | None = None,
        termination_aux: bool = False,
        termination_n_classes: int = 5,
        multi_offset_targets: list[int] | None = None,
    ):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout_p = float(dropout)
        self.use_checkpoint = use_checkpoint
        self.label_smoothing = float(label_smoothing)
        self.sep_id = sep_id
        self.tie_embeddings = bool(tie_embeddings)
        self.n_kv_head = n_kv_head if (n_kv_head is not None and n_kv_head > 0) else None
        self.use_sdpa = bool(use_sdpa)
        self.termination_aux = bool(termination_aux)
        self.termination_n_classes = int(termination_n_classes)
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, dropout, block_size, n_kv_head=self.n_kv_head, use_sdpa=self.use_sdpa) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        if self.tie_embeddings:
            self.head.weight = self.tok_emb.weight
        self.termination_head = (
            nn.Linear(n_embd, self.termination_n_classes)
            if self.termination_aux
            else None
        )

        self.multi_offset_targets = sorted(list(set([int(t) for t in multi_offset_targets]))) if multi_offset_targets else []
        self.offset_projs = nn.ModuleDict()
        if self.multi_offset_targets:
            for offset in self.multi_offset_targets:
                # 2-layer MLP with non-linear activation
                mlp = nn.Sequential(
                    nn.Linear(n_embd, n_embd),
                    nn.GELU(),
                    nn.Linear(n_embd, n_embd)
                )
                # Initialize both Linear layers to identity to preserve pretrained representations
                nn.init.eye_(mlp[0].weight)
                if mlp[0].bias is not None:
                    nn.init.zeros_(mlp[0].bias)
                nn.init.eye_(mlp[2].weight)
                if mlp[2].bias is not None:
                    nn.init.zeros_(mlp[2].bias)
                self.offset_projs[str(offset)] = mlp

        if loss_weights is not None:
            self.register_buffer("loss_weights", torch.tensor(loss_weights, dtype=torch.float32))
        else:
            self.register_buffer("loss_weights", torch.ones(vocab_size, dtype=torch.float32))

    def to_dict(self) -> dict:
        return {
            "vocab_size": int(self.vocab_size),
            "block_size": int(self.block_size),
            "n_layer": int(self.n_layer),
            "n_head": int(self.n_head),
            "n_embd": int(self.n_embd),
            "dropout": float(self.dropout_p),
            "sep_mask_enabled": self.sep_id is not None,
            "tie_embeddings": bool(self.tie_embeddings),
            "n_kv_head": self.n_kv_head,
            "use_sdpa": bool(self.use_sdpa),
            "termination_aux": bool(self.termination_aux),
            "termination_n_classes": int(self.termination_n_classes),
            "multi_offset_targets": self.multi_offset_targets,
        }

    def forward(self, idx, targets=None, return_aux: bool = False):
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
            seg_mask = (seg.unsqueeze(-1) == seg.unsqueeze(-2)).unsqueeze(1)  # (B,1,T,T)
            causal_mask = torch.tril(torch.ones(T, T, device=idx.device)).unsqueeze(0).unsqueeze(0) > 0  # (1,1,T,T)
            attn_mask = causal_mask & seg_mask  # Pre-combined mask (B,1,T,T)

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
        aux = {}
        if self.termination_head is not None:
            aux["termination_logits"] = self.termination_head(x)

        if self.offset_projs:
            offset_logits = {}
            for offset in self.multi_offset_targets:
                proj_x = self.offset_projs[str(offset)](x)
                offset_logits[offset] = self.head(proj_x)
            aux["offset_logits"] = offset_logits

        loss = None
        if targets is not None:
            # Check if all weights are 1.0 to avoid cross-entropy weight overhead / behavior divergence
            is_uniform = torch.all(self.loss_weights == 1.0).item()
            weight_to_use = None if is_uniform else self.loss_weights
            loss = F.cross_entropy(
                logits.float().view(-1, logits.size(-1)),
                targets.contiguous().view(-1),
                ignore_index=0,
                label_smoothing=self.label_smoothing,
                weight=weight_to_use,
            )
        if return_aux:
            return logits, loss, aux
        return logits, loss

class NoPropBlock(nn.Module):
    def __init__(self, n_embd, n_head, dropout, block_size, n_kv_head: int | None = None, use_sdpa: bool = False):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout, block_size, n_kv_head=n_kv_head, use_sdpa=use_sdpa)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
        self.denoise_head = nn.Linear(n_embd, n_embd)

    def forward(self, h, noisy_targets=None, attn_mask=None):
        if noisy_targets is not None:
            x = h + noisy_targets
        else:
            x = h

        x = x + self.attn(self.ln1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln2(x))

        pred_y = self.denoise_head(x)
        return x, pred_y

class NoPropTinyGPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_layer=3, n_head=4, n_embd=256, dropout=0.1, sep_id: int | None = 3, n_kv_head: int | None = None, use_sdpa: bool = False):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.sep_id = sep_id

        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            NoPropBlock(n_embd, n_head, dropout, block_size, n_kv_head=n_kv_head, use_sdpa=use_sdpa)
            for _ in range(n_layer)
        ])

        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

    def forward(self, idx, target_embeddings=None):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        h = self.tok_emb(idx) + self.pos_emb(pos)
        h = self.drop(h)

        attn_mask = None
        if self.sep_id is not None:
            sep = (idx == int(self.sep_id))
            seg = torch.cumsum(sep, dim=1)
            seg_mask = (seg.unsqueeze(-1) == seg.unsqueeze(-2)).unsqueeze(1) # (B,1,T,T)
            causal_mask = torch.tril(torch.ones(T, T, device=idx.device)).unsqueeze(0).unsqueeze(0) > 0 # (1,1,T,T)
            attn_mask = causal_mask & seg_mask # Pre-combined mask (B,1,T,T)

        preds = []
        for blk in self.blocks:
            h, pred_y = blk(h, noisy_targets=target_embeddings, attn_mask=attn_mask)
            preds.append(pred_y)

        h = self.ln_f(h)
        logits = self.head(h)
        return logits, preds

__all__ = ["TinyGPT", "NoPropBlock", "NoPropTinyGPT"]
