#!/usr/bin/env python3
"""
Zero-shot mutational fitness evaluation for CodonLM on:
1. Prokaryotic Protein DMS (Deep Mutational Scanning)
2. E. coli 5S rRNA DMS

Usage:
  python scripts/benchmark_zero_shot_mutations.py --run_id <RUN_ID>
  or
  python scripts/benchmark_zero_shot_mutations.py --run_dir <RUN_DIR>
"""

import argparse
import json
import os
import pandas as pd
import torch
from pathlib import Path
from scipy.stats import spearmanr

# Import our shared helpers
from scripts._shared import resolve_run, load_model, stoi as shared_stoi
from src.codonlm.codon_tokenize import to_ids, stoi as token_stoi

def score_sequence(model, token_ids, device):
    """
    Computes the log-likelihood of a sequence under the causal LM.
    """
    x = torch.tensor([token_ids], device=device).long()
    with torch.no_grad():
        logits, _ = model(x) # (1, T, V)
        # Shift targets and outputs
        # Target tokens are tokens[1:]
        # Output logits predicting them are at positions 0..T-2
        targets = x[0, 1:]
        pred_logits = logits[0, :-1, :]
        log_probs = torch.log_softmax(pred_logits, dim=-1)
        
        # Gather the log-probs of the actual targets
        gathered_log_probs = log_probs[torch.arange(len(targets)), targets]
        return float(gathered_log_probs.sum())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default=None)
    ap.add_argument("--run_dir", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    # 1. Resolve run directory and load metadata
    run_id, run_dir = resolve_run(run_id=args.run_id, run_dir=args.run_dir)
    print(f"[*] Running mutation scoring benchmark for run: {run_id}")

    # Determine device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[*] Using device: {device}")

    # Load Model (robustly handling consolidated vs legacy weight location)
    # Check if there is an explicit path in meta.json
    meta_path = run_dir / "meta.json"
    checkpoint_path = None
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
            checkpoint_path = meta.get("checkpoint_path")
            
    # Try loading the model
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"[*] Found checkpoint_path in meta: {checkpoint_path}")
        from scripts._shared import read_meta, ModelSpec, build_model
        meta = read_meta(run_dir)
        spec = ModelSpec.from_dict(meta["model_spec"])
        model = build_model(spec)
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)
        model.to(device).eval()
    else:
        # Fall back to load_model defaults
        model, spec = load_model(run_dir, device=device, ckpt_name="best.pt")

    # 2. Score Protein DMS Dataset
    protein_dms_path = Path("data/benchmarks/protein_dms.csv")
    protein_corr = 0.0
    if protein_dms_path.exists():
        print(f"[*] Loading protein DMS dataset from {protein_dms_path}...")
        df = pd.read_csv(protein_dms_path)
        
        wt_scores = {}
        mut_scores = []
        
        for idx, row in df.iterrows():
            wt_seq = row["wildtype_seq"]
            mut_seq = row["mutated_seq"]
            
            # Compute log likelihood for WT if not cached
            if wt_seq not in wt_scores:
                wt_ids = to_ids(wt_seq)
                wt_scores[wt_seq] = score_sequence(model, wt_ids, device)
                
            # Compute log likelihood for Mutant
            mut_ids = to_ids(mut_seq)
            mut_ll = score_sequence(model, mut_ids, device)
            
            delta_logp = mut_ll - wt_scores[wt_seq]
            mut_scores.append(delta_logp)
            
        df["predicted_delta_logp"] = mut_scores
        # Compute Spearman rank correlation
        protein_corr, p_val = spearmanr(df["predicted_delta_logp"], df["fitness_score"].astype(float))
        print(f"  - Protein DMS Spearman correlation: {protein_corr:.4f} (p={p_val:.4e})")
    else:
        print(f"[!] Protein DMS dataset not found at {protein_dms_path}")

    # 3. Score rRNA DMS Dataset
    rrna_dms_path = Path("data/benchmarks/rrna_dms.csv")
    rrna_corr = 0.0
    if rrna_dms_path.exists():
        print(f"[*] Loading E. coli 5S rRNA DMS dataset from {rrna_dms_path}...")
        df_rrna = pd.read_csv(rrna_dms_path)
        
        wt_scores = {}
        mut_scores = []
        
        for idx, row in df_rrna.iterrows():
            wt_seq = row["wildtype_seq"]
            mut_seq = row["mutated_seq"]
            
            # Compute log likelihood for WT if not cached
            if wt_seq not in wt_scores:
                wt_ids = to_ids(wt_seq)
                wt_scores[wt_seq] = score_sequence(model, wt_ids, device)
                
            # Compute log likelihood for Mutant
            mut_ids = to_ids(mut_seq)
            mut_ll = score_sequence(model, mut_ids, device)
            
            delta_logp = mut_ll - wt_scores[wt_seq]
            mut_scores.append(delta_logp)
            
        df_rrna["predicted_delta_logp"] = mut_scores
        # Compute Spearman rank correlation
        rrna_corr, p_val = spearmanr(df_rrna["predicted_delta_logp"], df_rrna["fitness_score"].astype(float))
        print(f"  - rRNA DMS Spearman correlation: {rrna_corr:.4f} (p={p_val:.4e})")
    else:
        print(f"[!] rRNA DMS dataset not found at {rrna_dms_path}")

    # Save scores to metrics.json
    scores_dir = run_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = scores_dir / "metrics.json"
    
    # Load existing metrics
    metrics = {}
    if metrics_path.exists():
        try:
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
        except Exception:
            pass
            
    # Update mutation benchmarking metrics
    metrics.update({
        "sota_protein_dms_spearman": float(protein_corr),
        "sota_rrna_dms_spearman": float(rrna_corr)
    })
    
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"[*] Wrote zero-shot mutation scoring results to {metrics_path}")

if __name__ == "__main__":
    main()
