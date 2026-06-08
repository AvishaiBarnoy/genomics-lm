import torch
import torch.nn as nn
from src.protein_lm.config import ProteinLMConfig, ProteinClassifierConfig
from src.protein_lm.models import ProteinConditionalTransformer

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

        # Multi-task heads
        self.heads = nn.ModuleDict({
            name: nn.Linear(config.n_embd, dim) 
            for name, dim in task_dims.items()
        })

    def forward(self, input_ids: torch.LongTensor) -> dict:
        """
        Returns a dictionary of logits for each task.
        """
        seq_length = input_ids.size(1)
        token_embeds = self.backbone.token_embedding(input_ids)
        pos_embeds = self.backbone.position_embedding(torch.arange(seq_length, device=input_ids.device))
        x = self.backbone.dropout(token_embeds + pos_embeds)

        # Use non-causal attention for classification tasks (bidirectional)
        # Note: Standard TransformerEncoderLayer uses bidirectional attention by default if src_mask=None
        for block in self.backbone.transformer_blocks:
            x = block(x)

        # Global Average Pooling (better for functional classification than just BOS)
        # Or just use BOS for structural tasks.
        bos_rep = x[:, 0, :]
        mean_rep = x.mean(dim=1)
        
        # Combine? Let's stick to mean-pooling for now.
        pooled = mean_rep

        return {name: head(pooled) for name, head in self.heads.items()}
