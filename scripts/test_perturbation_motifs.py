#!/usr/bin/env python3
"""
In-silico perturbation script to test if CodonLM has learned to associate
structural mRNA termination motifs (GC-rich hairpins, poly-T tracts) with termination cues.
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from scripts._shared import resolve_run, load_token_list, stoi

def run_perturbation_experiment(run_id: str, ckpt: str = "best.pt", n_prefixes: int = 50):
    run_id, run_dir = resolve_run(run_id=run_id)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[*] Loading model for run {run_id} on {device}...")
    
    from scripts._shared import read_meta, build_model, ModelSpec
    meta = read_meta(run_dir)
    spec_obj = ModelSpec.from_dict(meta["model_spec"])
    
    repo = Path(__file__).resolve().parents[1]
    weights_path = run_dir / "checkpoints" / ckpt
    if not weights_path.exists():
        weights_path = run_dir / ckpt
    if not weights_path.exists():
        weights_path = repo / "outputs" / "checkpoints" / run_id / ckpt
        
    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint {ckpt} not found for run {run_id}")
        
    ckpt_dict = torch.load(weights_path, map_location=device)
    state_dict = ckpt_dict["model"] if isinstance(ckpt_dict, dict) and "model" in ckpt_dict else ckpt_dict
    model = build_model(spec_obj)
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    
    tokens = load_token_list(run_dir)
    tok_stoi = stoi(tokens)
    
    # Identify termination tokens
    eos_id = tok_stoi.get("<EOS_CDS>")
    stop_ids = [tok_stoi[s] for s in ["TAA", "TAG", "TGA"] if s in tok_stoi]
    all_term_ids = ([eos_id] if eos_id is not None else []) + stop_ids
    
    # Load dataset to get prefixes
    manifest_path = run_dir / "combined_manifest.json"
    manifest = eval(manifest_path.read_text()) # safe load since it's a simple JSON/dict format
    dna_file_path = Path(manifest["datasets"][0]["dna"])
    if not dna_file_path.is_absolute():
        dna_file_path = repo / dna_file_path
        
    prefixes = []
    with open(dna_file_path) as f:
        for line in f:
            seq = line.strip().upper().replace("U", "T")
            if len(seq) >= 60:
                prefixes.append(seq[:30])  # 10 codons
            if len(prefixes) >= n_prefixes:
                break
                
    # Define perturbation tails (aligned to codon boundaries: multiples of 3 bp)
    # 1. Control tail: Neutral GC-balanced non-hairpin sequence
    # GCG (Ala), GCG (Ala), GCC (Ala), GCC (Ala), GCA (Ala), GCG (Ala), GCG (Ala), GCC (Ala), GCC (Ala), GCA (Ala) - 30bp
    control_tail = "GCGGCGGCCGCCGCAGCGGCGGCCGCCGCA"
    
    # 2. Poly-T tail: Runs of Ts
    # GCG (Ala), GCG (Ala), GCC (Ala), GCC (Ala), GCA (Ala), TTT (Phe), TTT (Phe), TTT (Phe), TTT (Phe), TTT (Phe) - 30bp
    poly_t_tail = "GCGGCGGCCGCCGCATTTTTTTTTTTTTTT"
    
    # 3. Hairpin tail: Stable GC-rich stem-loop structure
    # GCGGCCGC (Stem, 8bp), AAAAAA (Loop, 6bp), GCGGCCGC (Stem-complement, 8bp), GCGGCG (Spacer, 6bp) - 30bp
    # Codon aligned version:
    # GCG GCC GCG GAA AAA ACC GCG GCC GCG GCG (translates cleanly, forms GCGGCCGC-AAAAAA-GCGGCCGC stem-loop)
    hairpin_tail = "GCGGCCGCGGAAAAAACCGCGGCCGCGGCG"
    
    # 4. Full intrinsic terminator tail: Hairpin + Poly-T
    # GCG GCC GCG GAA AAA ACC GCG GCC GCT TTT (27 bp stem-loop + 3 bp Poly-T spacer)
    # Or to keep length 30 bp:
    # GCC GCG GCC GCG AAA ACC GCG GCC GCT TTT (forms GCGGCCGC stem with AAAA loop followed by TTTTTT)
    # Let's write a clean codon version:
    # GCC GCG GCC GCG AAA ACC GCG GCC GCT TTT
    # Stem: GCCGCGGCC (9bp) - Loop: GAAAAC (6bp) - Stem: GCCGCGGCC (9bp) - U-tract: TTTTTT (6bp)
    # Wait, GCC GCG GCC GCG AAA ACC GCG GCC GCT TTT has:
    # GCC GCG GCC GCG (12bp) - AAA ACC (6bp) - GCG GCC GCT TTT (12bp)
    # Total 30 bp.
    terminator_tail = "GCCGCGGCCGCGAAAACCGCGGCCGCTTTT"
    
    variants = {
        "Control (Neutral)": control_tail,
        "Poly-T Tract": poly_t_tail,
        "Hairpin Stem-Loop": hairpin_tail,
        "Full Terminator (Hairpin + Poly-T)": terminator_tail
    }
    
    results = {k: [] for k in variants}
    eos_probs = {k: [] for k in variants}
    stop_probs = {k: [] for k in variants}
    
    bos_id = tok_stoi.get("<BOS_CDS>")
    
    print(f"[*] Evaluated next-token probabilities on {n_prefixes} prefixes:")
    for name, tail in variants.items():
        for pref in prefixes:
            full_seq = pref + tail
            
            # Tokenize sequence
            ctx_ids = []
            if bos_id is not None:
                ctx_ids.append(bos_id)
            for i in range(0, len(full_seq), 3):
                codon = full_seq[i:i+3]
                if codon in tok_stoi:
                    ctx_ids.append(tok_stoi[codon])
                    
            # Compute logits
            x = torch.tensor([ctx_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                logits, _ = model(x)
                # Next token logits
                next_logits = logits[0, -1]
                probs = F.softmax(next_logits, dim=-1)
                
            # Get probability of termination
            term_prob = sum(probs[tid].item() for tid in all_term_ids)
            eos_prob = probs[eos_id].item() if eos_id is not None else 0.0
            stop_prob = sum(probs[sid].item() for sid in stop_ids)
            
            results[name].append(term_prob)
            eos_probs[name].append(eos_prob)
            stop_probs[name].append(stop_prob)
            
    print("\n=== In-Silico Perturbation Results ===")
    for name in variants:
        mean_term = np.mean(results[name])
        mean_eos = np.mean(eos_probs[name])
        mean_stop = np.mean(stop_probs[name])
        print(f"{name}:")
        print(f"  - Total Termination Probability (Stop or EOS): {mean_term*100:.4f}%")
        print(f"  - EOS (<EOS_CDS>) Probability: {mean_eos*100:.4f}%")
        print(f"  - Stop Codon Probability: {mean_stop*100:.4f}%")
        
    # Write report JSON
    out_dir = run_dir / "motif_mining"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_data = {
        "termination_probabilities": {k: float(np.mean(results[k])) for k in variants},
        "eos_probabilities": {k: float(np.mean(eos_probs[k])) for k in variants},
        "stop_probabilities": {k: float(np.mean(stop_probs[k])) for k in variants},
    }
    import json
    with open(out_dir / "perturbation_motifs_analysis.json", "w") as f:
        json.dump(report_data, f, indent=4)
    print(f"\n[success] Perturbation report saved to {out_dir / 'perturbation_motifs_analysis.json'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="2026-06-06_stage2.5_6L4H_d256_e20")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--prefixes", type=int, default=50)
    args = ap.parse_args()
    
    run_perturbation_experiment(args.run_id, args.ckpt, args.prefixes)
