import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from src.protein_lm.config import ProteinLMConfig, ProteinClassifierConfig
from src.protein_lm.models import ProteinConditionalTransformer

class AttentionPooling(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_embd))
        self.key_proj = nn.Linear(n_embd, n_embd)
        self.value_proj = nn.Linear(n_embd, n_embd)
        nn.init.normal_(self.query, std=0.02)

    def forward(self, x, attention_mask=None):
        # x shape: (B, T, D)
        k = self.key_proj(x)  # (B, T, D)
        v = self.value_proj(x)  # (B, T, D)
        
        q = self.query.view(1, -1, 1)  # (1, D, 1)
        attn_logits = torch.matmul(k, q).squeeze(-1)  # (B, T)
        attn_logits = attn_logits / (k.size(-1) ** 0.5)
        
        if attention_mask is not None:
            attn_logits = attn_logits.masked_fill(attention_mask == 0, float('-inf'))
            
        attn_weights = torch.softmax(attn_logits, dim=-1)  # (B, T)
        pooled = torch.matmul(attn_weights.unsqueeze(1), v).squeeze(1)  # (B, D)
        return pooled, attn_weights

class MultiTaskProteinClassifier(nn.Module):
    """
    A multi-task classifier for proteins, predicting family, stability, and function.
    Uses a ProteinLM backbone.
    """
    def __init__(self, config: ProteinClassifierConfig, task_dims: dict):
        """
        Args:
            config: Model configuration.
            task_dims: Dictionary mapping task name (e.g., 'family', 'stability') 
                       to number of classes for that task.
        """
        super().__init__()
        self.config = config
        self.task_dims = task_dims

        # Backbone
        self.backbone = ProteinConditionalTransformer(
            ProteinLMConfig(
                vocab_size=config.vocab_size,
                n_layer=config.n_layer,
                n_head=config.n_head,
                n_embd=config.n_embd,
                block_size=config.block_size,
                dropout=config.dropout,
            )
        )

        # Attention-based or standard mean pooling
        self.pooling_type = getattr(config, "pooling", "mean")
        if self.pooling_type == "attention":
            self.pooler = AttentionPooling(config.n_embd)
        else:
            self.pooler = None

        # Shared Latent Bottleneck Layer (Phase 2.5)
        self.shared_latent = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd),
            nn.LayerNorm(config.n_embd),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )

        # Multi-task heads branched off the shared bottleneck representation
        self.heads = nn.ModuleDict({
            name: nn.Linear(config.n_embd, dim) 
            for name, dim in task_dims.items()
        })

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict:
        """
        Returns a dictionary of logits for each task.
        """
        seq_length = input_ids.size(1)
        token_embeds = self.backbone.token_embedding(input_ids)
        pos_embeds = self.backbone.position_embedding(torch.arange(seq_length, device=input_ids.device))
        x = self.backbone.dropout(token_embeds + pos_embeds)

        # Causal vs Bidirectional configuration (Phase 1)
        is_causal = not getattr(self.config, "bidirectional", True)
        if is_causal:
            causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_length, device=input_ids.device)
        else:
            causal_mask = None

        use_checkpoint = self.config.use_checkpoint if hasattr(self.config, "use_checkpoint") else False
        if use_checkpoint and self.training:
            for block in self.backbone.transformer_blocks:
                try:
                    # Pass use_reentrant=False to silence deprecation warnings in newer PyTorch
                    x = checkpoint(
                        block,
                        x,
                        src_mask=causal_mask,
                        src_key_padding_mask=(attention_mask == 0) if attention_mask is not None else None,
                        use_reentrant=False,
                    )
                except TypeError:
                    x = checkpoint(block, x, src_mask=causal_mask)
        else:
            for block in self.backbone.transformer_blocks:
                x = block(
                    x,
                    src_mask=causal_mask,
                    src_key_padding_mask=(attention_mask == 0) if attention_mask is not None else None,
                )

        attn_weights = None
        if self.pooling_type == "attention":
            pooled, attn_weights = self.pooler(x, attention_mask=attention_mask)
        else:
            if attention_mask is None:
                pooled = x.mean(dim=1)
            else:
                mask = attention_mask.to(dtype=x.dtype, device=x.device).unsqueeze(-1)
                pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        # Project through the shared latent layer bottleneck
        latent = self.shared_latent(pooled)

        # Branch classifier heads off the shared bottleneck representation
        logits_dict = {name: head(latent) for name, head in self.heads.items()}
        if attn_weights is not None:
            logits_dict["attention_weights"] = attn_weights

        return logits_dict
