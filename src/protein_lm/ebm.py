import torch
import torch.nn as nn

class ProteinLatentEBM(nn.Module):
    """
    Continuous Latent-Space Energy-Based Model (EBM) for Proteins.
    Accepts mean-pooled or bottleneck latent embeddings z and outputs a single scalar energy score.
    """
    def __init__(self, n_embd: int = 256, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Continuous latent tensor of shape (batch_size, n_embd) or (batch_size, seq_len, n_embd)
        
        Returns:
            Scalar energy score of shape (batch_size,)
        """
        if z.ndim == 3:
            z = z.mean(dim=1)
        return self.net(z).squeeze(-1)
