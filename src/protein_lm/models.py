import torch
import torch.nn as nn
from src.protein_lm.config import ProteinLMConfig, ProteinClassifierConfig

class ProteinConditionalTransformer(nn.Module):
    """
    A GPT-style causal transformer for conditional protein language modeling.
    """
    def __init__(self, config: ProteinLMConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

        # Using TransformerEncoderLayer which is a standard building block.
        # We will apply the causal mask manually in the forward pass.
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.n_embd,
                nhead=config.n_head,
                dim_feedforward=4 * config.n_embd,
                dropout=config.dropout,
                batch_first=True,
                activation='gelu' # Commonly used in transformers
            ) for _ in range(config.n_layer)
        ])

        self.layer_norm = nn.LayerNorm(config.n_embd)
        self.output_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """
        Forward pass for the language model.

        Args:
            input_ids: A tensor of shape (batch_size, seq_length).

        Returns:
            Logits of shape (batch_size, seq_length, vocab_size).
        """
        seq_length = input_ids.size(1)

        # Token and position embeddings
        token_embeds = self.token_embedding(input_ids)
        pos_embeds = self.position_embedding(torch.arange(seq_length, device=input_ids.device))
        x = self.dropout(token_embeds + pos_embeds)

        # Causal mask to ensure that the model doesn't cheat by looking ahead
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_length, device=input_ids.device)

        for block in self.transformer_blocks:
            x = block(x, src_mask=causal_mask)

        x = self.layer_norm(x)
        logits = self.output_head(x)

        return logits

class ProteinClassifier(nn.Module):
    """
    A protein classifier that uses the language model as a backbone.
    """
    def __init__(self, config: ProteinClassifierConfig):
        super().__init__()
        self.config = config

        # The backbone is the language model, which will act as a feature extractor.
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

        self.classification_head = nn.Linear(config.n_embd, config.num_classes)

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """
        Forward pass for the classifier.

        Args:
            input_ids: A tensor of shape (batch_size, seq_length).

        Returns:
            Logits of shape (batch_size, num_classes).
        """
        # To get the features, we need the hidden states from the backbone.
        # We can't just call the backbone's forward method, as it returns logits.
        # Instead, we'll run the components of the backbone's forward pass here.
        seq_length = input_ids.size(1)

        token_embeds = self.backbone.token_embedding(input_ids)
        pos_embeds = self.backbone.position_embedding(torch.arange(seq_length, device=input_ids.device))
        x = self.backbone.dropout(token_embeds + pos_embeds)

        # We don't need a causal mask for the classifier, as we want to use the whole sequence.
        # However, to reuse the backbone, we are using the same forward pass.
        # For classification, it is common to use a non-causal mask, but for simplicity
        # and to reuse the code, we will use the causal mask.
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_length, device=input_ids.device)

        for block in self.backbone.transformer_blocks:
            x = block(x, src_mask=causal_mask)

        # We pool the representation of the [BOS] token, which is at the first position.
        # This is a common technique to get a fixed-size representation of the sequence.
        bos_representation = x[:, 0, :]

        logits = self.classification_head(bos_representation)
        return logits