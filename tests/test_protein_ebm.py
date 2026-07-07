import torch
import numpy as np
from pathlib import Path

from src.protein_lm.tokenizer import ProteinTokenizer
from src.protein_lm.ebm import ProteinLatentEBM
from src.protein_lm.config import ProteinClassifierConfig
from src.protein_lm.models_multi import MultiTaskProteinClassifier
from src.protein_lm.sampler import latent_langevin_sample
from src.protein_lm.train_ebm import corrupt_sequence
from scripts.generative_design_loop import compute_shannon_entropy

def test_corrupt_sequence():
    seq = "MGEKVALVIA"
    mutated = corrupt_sequence(seq, mutation_rate=0.30)
    assert len(mutated) == len(seq)
    # Check that at least some characters mutated
    assert mutated != seq


def test_shannon_entropy_loop_detection():
    # Repetitive babble has zero/low entropy
    rep = ["ATG"] * 15
    entropy_rep = compute_shannon_entropy(rep)
    assert entropy_rep < 1.0

    # Complex patterns have high entropy
    diverse = ["ATG", "AAA", "TGC", "TGG", "TTT", "CCA", "CCT", "CGT", "GTA", "GCT"] * 2
    entropy_div = compute_shannon_entropy(diverse)
    assert entropy_div > 1.5


def test_ebm_and_langevin_optimization():
    # 1. Setup mock classifier configuration
    tokenizer = ProteinTokenizer()
    model_cfg = ProteinClassifierConfig(
        vocab_size=len(tokenizer),
        n_layer=2,
        n_head=2,
        n_embd=64,
        block_size=128,
        dropout=0.1,
        pooling="mean",
        num_classes=2,
    )
    task_dims = {"family": 10, "stability": 2}
    
    critic = MultiTaskProteinClassifier(model_cfg, task_dims)
    ebm = ProteinLatentEBM(n_embd=64, hidden_dim=128)
    
    # 2. Run Langevin optimization on a mock sequence
    initial_seq = "MAPKVALVIA"
    opt_seq, energy_history = latent_langevin_sample(
        ebm_model=ebm,
        critic_model=critic,
        tokenizer=tokenizer,
        initial_seq=initial_seq,
        steps=10,
        lr=0.1,
        noise_std=0.0,  # disable noise to check strict gradient convergence
        device=torch.device("cpu")
    )
    
    # Check shape alignment and decoding output
    assert len(opt_seq) == len(initial_seq)
    assert len(energy_history) == 10
    
    # In deterministic gradient descent (without noise), the energy should decrease over steps
    assert energy_history[-1] <= energy_history[0], "Energy did not decrease during Langevin optimization"
