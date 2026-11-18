#!/usr/bin/env python3
"""
Diagnostic and Weight-Space Programming script for CodonLM.
Inspired by 'Sparks of Weight-Space Programming' (Takehi et al., 2026).
Inspects, diagnoses, and directly modifies neural network weights without gradient descent.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from src.codonlm.checkpoints import build_codon_model_from_cfg, load_codon_checkpoint


def load_model_and_metadata(run_dir: Path | str, device: torch.device = torch.device("cpu")):
    run_dir = Path(run_dir)
    state, cfg, _ = load_codon_checkpoint(run_dir)
    model = build_codon_model_from_cfg(cfg)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    itos_path = run_dir / "itos.txt"
    if not itos_path.exists():
        itos_path = Path("data/processed/corrected/corrected-codonlm-v1/genome/itos.txt")
    with open(itos_path) as f:
        itos = [line.strip() for line in f]
    stoi = {tok: i for i, tok in enumerate(itos)}
    return model, cfg, itos, stoi


def diagnose_weights(model, itos, stoi):
    """Diagnose unembeddings, stop codon representations, and layer directions."""
    head_w = model.head.weight.detach().cpu() # [68, 384]
    tok_w = model.tok_emb.weight.detach().cpu() # [68, 384]
    ln_f_w = model.ln_f.weight.detach().cpu() # [384]
    ln_f_b = model.ln_f.bias.detach().cpu() # [384]

    stop_tokens = ["TAA", "TAG", "TGA"]
    stop_ids = [stoi[c] for c in stop_tokens if c in stoi]
    eos_id = stoi.get("<EOS_CDS>")
    
    # Check stop vs normal codon norms
    normal_ids = [i for i, tok in enumerate(itos) if len(tok) == 3 and set(tok) <= set("ACGT") and tok not in stop_tokens]
    
    stop_head_norms = head_w[stop_ids].norm(dim=-1).tolist()
    normal_head_norms = head_w[normal_ids].norm(dim=-1).mean().item()
    
    print(f"--- Weight Diagnosis ---")
    print(f"Normal codon avg head norm: {normal_head_norms:.4f}")
    for tok, sid in zip(stop_tokens, stop_ids):
        print(f"Stop codon {tok} (id {sid}) head norm: {head_w[sid].norm().item():.4f}")
    if eos_id is not None:
        print(f"<EOS_CDS> (id {eos_id}) head norm: {head_w[eos_id].norm().item():.4f}")

    # Inspect cosine similarity between stop codons and other codons
    # Check if stop codon head vectors are orthogonal or anti-correlated to the GC-rich subspace
    gc_heavy = [i for i, tok in enumerate(itos) if len(tok) == 3 and set(tok) <= set("ACGT") and tok.count("G") + tok.count("C") == 3]
    gc_light = [i for i, tok in enumerate(itos) if len(tok) == 3 and set(tok) <= set("ACGT") and tok.count("A") + tok.count("T") == 3]
    
    gc_heavy_mean_dir = F.normalize(head_w[gc_heavy].mean(dim=0), dim=0)
    gc_light_mean_dir = F.normalize(head_w[gc_light].mean(dim=0), dim=0)
    stop_mean_dir = F.normalize(head_w[stop_ids].mean(dim=0), dim=0)
    
    print(f"Cos sim (Stop mean dir, GC-heavy mean dir): {torch.dot(stop_mean_dir, gc_heavy_mean_dir).item():.4f}")
    print(f"Cos sim (Stop mean dir, AT-heavy mean dir): {torch.dot(stop_mean_dir, gc_light_mean_dir).item():.4f}")

    # Inspect unigram logit bias: head has no bias!
    # Because head has no bias (head = nn.Linear(384, 68, bias=False)),
    # the model projects ln_f(x) directly onto head_w: logits = x @ head_w.T.
    # Note that ln_f(x) has a mean of ln_f_b (bias)!
    base_logits_from_bias = (ln_f_b @ head_w.T).tolist()
    print(f"Base logits induced by ln_f.bias on stop tokens:")
    for tok, sid in zip(stop_tokens, stop_ids):
        print(f"  {tok}: {base_logits_from_bias[sid]:.4f}")
    print(f"Base logits induced by ln_f.bias on normal codons (mean): {np.mean([base_logits_from_bias[i] for i in normal_ids]):.4f}")
    print(f"Base logits induced by ln_f.bias on GC-heavy codons (mean): {np.mean([base_logits_from_bias[i] for i in gc_heavy]):.4f}")

    return {
        "normal_head_norms": normal_head_norms,
        "stop_head_norms": dict(zip(stop_tokens, stop_head_norms)),
        "base_logits_stops": {tok: base_logits_from_bias[sid] for tok, sid in zip(stop_tokens, stop_ids)},
        "base_logits_normal": float(np.mean([base_logits_from_bias[i] for i in normal_ids])),
        "base_logits_gc_heavy": float(np.mean([base_logits_from_bias[i] for i in gc_heavy])),
    }


if __name__ == "__main__":
    run_dir = "runs/corrected-multi-offset-heads-seed1337"
    model, cfg, itos, stoi = load_model_and_metadata(run_dir)
    diagnose_weights(model, itos, stoi)
