
import torch
import torch.nn as nn
from torch.nn import functional as F
from src.protein_lm.config import ProteinLMConfig, ProteinClassifierConfig

class ProteinConditionalTransformer(nn.Module):
    def __init__(self, config: ProteinLMConfig):
        super().__init__()
        self.config = config
        
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.n_embd, 
                nhead=config.n_head,
                dim_feedforward=4 * config.n_embd,
                dropout=config.dropout,
                batch_first=True,
                # Causal attention is not a standard option in TransformerEncoderLayer
                # so we will apply the mask manually in the forward pass.
            ) for _ in range(config.n_layer)
        ])
        
        self.layer_norm = nn.LayerNorm(config.n_embd)
        self.output_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, input_ids: torch.LongTensor):
        seq_length = input_ids.size(1)
        
        # Token and position embeddings
        token_embeds = self.token_embedding(input_ids)
        pos_embeds = self.position_embedding(torch.arange(seq_length, device=input_ids.device))
        x = self.dropout(token_embeds + pos_embeds)
        
        # Causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_length, device=input_ids.device)

        for block in self.transformer_blocks:
            x = block(x, src_mask=causal_mask)
            
        x = self.layer_norm(x)
        logits = self.output_head(x)
        
        return logits

class ProteinClassifier(nn.Module):
    def __init__(self, config: ProteinClassifierConfig):
        super().__init__()
        self.config = config
        
        # We can reuse the LM backbone
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

    def forward(self, input_ids: torch.LongTensor):
        # The forward pass of the backbone returns logits, we need the hidden states
        # Let's modify the backbone to return hidden states.
        # For now, let's just grab the embeddings and process them.
        
        seq_length = input_ids.size(1)
        
        token_embeds = self.backbone.token_embedding(input_ids)
        pos_embeds = self.backbone.position_embedding(torch.arange(seq_length, device=input_ids.device))
        x = self.backbone.dropout(token_embeds + pos_embeds)
        
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_length, device=input_ids.device)

        for block in self.backbone.transformer_blocks:
            x = block(x, src_mask=causal_mask)
            
        # Pool the representation - we will use the hidden state of the BOS token
        bos_representation = x[:, 0, :]
        
        logits = self.classification_head(bos_representation)
        return logits
