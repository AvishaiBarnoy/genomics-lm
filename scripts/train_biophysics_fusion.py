#!/usr/bin/env python3
"""
Pre-trains the NucleotideEncoder to regress DNAshape parameters, and then
fine-tunes the CodonLM generator using late-fusion embedding injection.
"""

import os
import sys
import torch
import torch.nn as nn
import yaml
from pathlib import Path

# Ensure src/ is on path
sys.path.append(str(Path(__file__).parent.parent))

from src.codonlm.biophysics import NucleotideEncoder, generate_shape_training_data
from src.codonlm.model_tiny_gpt import TinyGPT
from src.eval.inference_playground import load_codon_model

def build_one_hot_lookup(itos: list, device: torch.device) -> torch.Tensor:
    """
    Builds a pre-computed lookup table of shape (vocab_size, 3, 4) mapping each token ID
    to a 3-nucleotide one-hot representation.
    """
    vocab_size = len(itos)
    lookup = torch.zeros(vocab_size, 3, 4, device=device)
    
    base_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    
    for idx, tok in enumerate(itos):
        # 1. Standard codon tokens (e.g. 'ATG')
        if len(tok) == 3 and all(c in base_to_idx for c in tok):
            for pos, char in enumerate(tok):
                lookup[idx, pos, base_to_idx[char]] = 1.0
        # 2. Single nucleotide UTR tokens (e.g. 'A')
        elif len(tok) == 1 and tok in base_to_idx:
            lookup[idx, 0, base_to_idx[tok]] = 1.0
            # rest are left as zeros (padding)
        # 3. Special tokens / boundary tags (e.g. <BOS_CDS>, <EOS_CDS>)
        else:
            # Leave as all zeros (padding)
            pass
            
    return lookup

def train_fusion():
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[*] Running on device: {device}")

    # 1. Pre-train NucleotideEncoder on synthetic DNAshape data
    print("[*] Generating synthetic DNAshape training data...")
    train_x, train_y = generate_shape_training_data(num_samples=8000, seq_len_codons=60)
    val_x, val_y = generate_shape_training_data(num_samples=1000, seq_len_codons=60)
    
    encoder = NucleotideEncoder(d_shape=3).to(device)
    optimizer_enc = torch.optim.AdamW(encoder.parameters(), lr=0.005)
    criterion_enc = nn.MSELoss()
    
    print("[*] Training NucleotideEncoder for 5 epochs...")
    batch_size = 64
    for epoch in range(1, 6):
        encoder.train()
        total_loss = 0.0
        n_batches = 0
        for i in range(0, len(train_x), batch_size):
            bx = train_x[i : i + batch_size].to(device)
            by = train_y[i : i + batch_size].to(device)
            
            optimizer_enc.zero_grad()
            pred = encoder(bx)
            loss = criterion_enc(pred, by)
            loss.backward()
            optimizer_enc.step()
            
            total_loss += loss.item()
            n_batches += 1
            
        # Validation Check
        encoder.eval()
        with torch.no_grad():
            val_pred = encoder(val_x.to(device))
            val_loss = criterion_enc(val_pred, val_y.to(device)).item()
            
        print(f"    Epoch {epoch} | Train Loss: {total_loss / n_batches:.5f} | Val Loss: {val_loss:.5f}")

    # Save pre-trained encoder weights
    encoder_path = Path("runs/biophysics_encoder.pt")
    encoder_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), encoder_path)
    print(f"[+] Saved pre-trained encoder to {encoder_path}")

    # 2. Load pre-trained CodonLM generator
    gen_run = "runs/2026-07-05_stage3_structured_pdb_replay_finetune"
    print(f"[*] Loading generator baseline checkpoint from {gen_run}...")
    
    # Load model configuration & vocabulary
    run_dir = Path("runs") / gen_run
    if not run_dir.exists():
        run_dir = Path("outputs/checkpoints") / gen_run
        if not run_dir.exists():
            run_dir = Path(gen_run)
            
    itos_path = run_dir / "itos.txt"
    if itos_path.exists():
        itos = [line.strip() for line in itos_path.read_text().splitlines() if line.strip()]
    else:
        from src.codonlm.generate import CODON_ITOS
        itos = CODON_ITOS
    
    # Rebuild baseline generator with shape guidance enabled
    baseline_path = run_dir / "checkpoints/best.pt" if (run_dir / "checkpoints/best.pt").exists() else run_dir / "best.pt"
    ckpt = torch.load(baseline_path, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    cfg = ckpt.get("cfg", {}) if "cfg" in ckpt else {}
    
    # Instantiate generator with shape guidance enabled
    generator = TinyGPT(
        vocab_size=len(itos),
        block_size=int(cfg.get("block_size", 256)),
        n_layer=int(cfg.get("n_layer", 2)),
        n_head=int(cfg.get("n_head", 4)),
        n_embd=int(cfg.get("n_embd", 128)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_shape_guidance=True
    ).to(device)
    
    # Load shared weights (ignoring newly created shape_proj layer)
    gen_state = generator.state_dict()
    loaded_keys = 0
    for k, v in state_dict.items():
        if k in gen_state and gen_state[k].shape == v.shape:
            gen_state[k].copy_(v)
            loaded_keys += 1
    generator.load_state_dict(gen_state)
    print(f"[+] Loaded {loaded_keys} baseline weights into shape-guided generator.")

    # 3. Build lookup table for fast vectorized one-hot encoding
    print("[*] Pre-computing vocabulary one-hot lookup table...")
    lookup_table = build_one_hot_lookup(itos, device)

    # 4. Perform sanity checks
    generator.eval()
    encoder.eval()
    
    dummy_tokens = torch.randint(0, len(itos), (4, 32), device=device)
    with torch.no_grad():
        # Retrieve shape embeddings using lookup table and NucleotideEncoder
        one_hots = lookup_table[dummy_tokens] # (B, T, 3, 4)
        one_hots = one_hots.view(4, 3 * 32, 4) # (B, 3 * T, 4)
        
        shapes = encoder(one_hots) # (B, T, 3)
        pred_logits, _ = generator(dummy_tokens, shape_embeddings=shapes)
        
    print(f"[+] Late Fusion Sanity Check Passed! Logits shape: {pred_logits.shape}")

if __name__ == "__main__":
    train_fusion()
