import torch
import pytest
from src.codonlm.biophysics import NucleotideEncoder, generate_shape_training_data
from src.codonlm.model_tiny_gpt import TinyGPT
from scripts.train_biophysics_fusion import build_one_hot_lookup

def test_biophysics_encoder_and_fusion():
    # 1. Test NucleotideEncoder shapes
    encoder = NucleotideEncoder(d_shape=3)
    encoder.eval()
    
    # Input has shape (B, 3L, 4), where L = 20, so 3L = 60
    bx = torch.zeros(4, 60, 4)
    pred_shapes = encoder(bx)
    assert pred_shapes.shape == (4, 20, 3)

    # 2. Test synthetic training data generation
    train_x, train_y = generate_shape_training_data(num_samples=10, seq_len_codons=15)
    assert train_x.shape == (10, 45, 4)
    assert train_y.shape == (10, 15, 3)

    # 3. Test lookup table mapping
    itos = ["ATG", "A", "<BOS_CDS>", "<PAD_CDS>"]
    lookup = build_one_hot_lookup(itos, device=torch.device("cpu"))
    assert lookup.shape == (4, 3, 4)
    
    # Codon 'ATG' should be fully encoded: A=idx 0, T=idx 3, G=idx 2
    assert lookup[0, 0, 0] == 1.0 # A
    assert lookup[0, 1, 3] == 1.0 # T
    assert lookup[0, 2, 2] == 1.0 # G
    
    # Single nucleotide 'A' should be encoded at index 0, followed by zeros
    assert lookup[1, 0, 0] == 1.0 # A
    assert (lookup[1, 1] == 0.0).all()
    assert (lookup[1, 2] == 0.0).all()

    # Special token should be all zeros
    assert (lookup[2] == 0.0).all()

    # 4. Test generator embedding injection
    generator = TinyGPT(
        vocab_size=len(itos),
        block_size=64,
        n_layer=1,
        n_head=1,
        n_embd=16,
        use_shape_guidance=True
    )
    generator.eval()

    dummy_tokens = torch.randint(0, len(itos), (2, 10))
    one_hots = lookup[dummy_tokens] # (2, 10, 3, 4)
    one_hots = one_hots.view(2, 30, 4)
    
    shapes = encoder(one_hots) # (2, 10, 3)
    logits, _ = generator(dummy_tokens, shape_embeddings=shapes)
    assert logits.shape == (2, 10, 4)
