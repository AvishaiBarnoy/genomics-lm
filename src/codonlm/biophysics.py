import torch
import torch.nn as nn
import random
from typing import List, Tuple
from scripts.probe_structural_awareness import get_theoretical_shape

class NucleotideEncoder(nn.Module):
    """
    Lightweight 1D CNN that compresses raw one-hot DNA nucleotide sequences
    of length 3L to codon-aligned biophysical shapes of length L.
    """
    def __init__(self, d_shape: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(32, d_shape, kernel_size=3, stride=3, padding=0)
        )

    def forward(self, one_hot: torch.Tensor) -> torch.Tensor:
        """
        Args:
            one_hot: Tensor of shape (B, 3L, 4) containing one-hot nucleotide values.
            
        Returns:
            predicted_shapes: Tensor of shape (B, L, d_shape) representing predicted shapes.
        """
        # Convert (B, 3L, 4) -> (B, 4, 3L) for PyTorch Conv1d
        x = one_hot.transpose(1, 2)
        out = self.net(x) # (B, d_shape, L)
        # Convert (B, d_shape, L) -> (B, L, d_shape)
        return out.transpose(1, 2)

def generate_shape_training_data(num_samples: int = 5000, seq_len_codons: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates synthetic DNA sequences and computes their theoretical shape targets
    (MGW, Roll, EP) averaged over each codon.
    """
    bases = ["A", "C", "G", "T"]
    base_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    
    seq_len_nt = 3 * seq_len_codons
    one_hots = []
    targets = []
    
    for _ in range(num_samples):
        # Generate random DNA sequence
        seq = "".join(random.choice(bases) for _ in range(seq_len_nt))
        
        # One-hot encoding
        oh = torch.zeros(seq_len_nt, 4)
        for idx, base in enumerate(seq):
            oh[idx, base_to_idx[base]] = 1.0
        one_hots.append(oh)
        
        # Calculate theoretical shape (14 properties, extract MGW, Roll, EP)
        shapes_dict = get_theoretical_shape(seq)
        mgw = shapes_dict["MGW"]
        roll = shapes_dict["Roll"]
        ep = shapes_dict["EP"]
        
        # Convert to tensor of shape (3L, 3)
        nt_shapes = torch.stack([
            torch.tensor(mgw, dtype=torch.float32),
            torch.tensor(roll, dtype=torch.float32),
            torch.tensor(ep, dtype=torch.float32)
        ], dim=-1)
        
        # Average shape values within each codon
        codon_shapes = nt_shapes.view(seq_len_codons, 3, 3).mean(dim=1)
        targets.append(codon_shapes)
        
    return torch.stack(one_hots), torch.stack(targets)
