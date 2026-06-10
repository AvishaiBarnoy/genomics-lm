#!/usr/bin/env python3
"""
Gene essentiality evaluation using linear probing on extracted sequence embeddings:
1. Lambda Phage Essentiality (Piya et al., 2023)
2. Pseudomonas aeruginosa Essentiality (Turner et al., 2015)

Usage:
  python -m scripts.benchmark_gene_essentiality --run_id <RUN_ID>
"""

import argparse
import json
import os
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

# Import shared helpers
from scripts._shared import resolve_run, load_model
from src.codonlm.codon_tokenize import to_ids
from scripts.extract_embeddings import _pool_hidden, _dna_to_codon_tokens

def extract_embeddings_for_seqs(model, seqs, stoi, bos_id, eos_id, pad_id, block_size, device):
    """
    Extracts mean-pooled sequence embeddings from CodonLM backbone for a list of sequences.
    """
    embeddings = []
    with torch.no_grad():
        for seq in seqs:
            codons = _dna_to_codon_tokens(seq)
            toks = []
            if bos_id is not None:
                toks.append(bos_id)
            for c in codons:
                if c in stoi:
                    toks.append(stoi[c])
            if eos_id is not None:
                toks.append(eos_id)
                
            if not toks:
                # Fallback to zeros if empty sequence
                embeddings.append(np.zeros(model.tok_emb.embedding_dim))
                continue
                
            ids_tensor = torch.tensor(
                toks[:block_size], dtype=torch.long, device=device
            ).unsqueeze(0)
            nonpad = ids_tensor.ne(pad_id)
            pooled = _pool_hidden(model, ids_tensor, nonpad)
            embeddings.append(pooled.squeeze(0).cpu().numpy())
            
    return np.stack(embeddings, axis=0)

def run_linear_probe(X, y):
    """
    Evaluates linear probe performance using stratified 5-fold cross-validation.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs = []
    f1s = []
    mccs = []
    
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        clf.fit(X_train, y_train)
        
        preds = clf.predict(X_test)
        
        accs.append(accuracy_score(y_test, preds))
        f1s.append(f1_score(y_test, preds, average='binary', zero_division=0))
        mccs.append(matthews_corrcoef(y_test, preds))
        
    return {
        "acc": float(np.mean(accs)),
        "f1": float(np.mean(f1s)),
        "mcc": float(np.mean(mccs))
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default=None)
    ap.add_argument("--run_dir", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    # 1. Resolve run directory and load metadata
    run_id, run_dir = resolve_run(run_id=args.run_id, run_dir=args.run_dir)
    print(f"[*] Running gene essentiality benchmark for run: {run_id}")

    # Determine device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[*] Using device: {device}")

    # Load Model (robustly handling consolidated vs legacy weight location)
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
        model, spec = load_model(run_dir, device=device, ckpt_name="best.pt")

    # Load vocabulary mappings
    from scripts import query_model as Q
    itos, stoi = Q._load_vocab(run_dir)
    
    bos_id = stoi.get("<BOS_CDS>")
    eos_id = stoi.get("<EOS_CDS>")
    pad_id = stoi.get("<PAD>", 0)
    block_size = spec.block_size if hasattr(spec, 'block_size') else 512

    # 2. Lambda Phage Gene Essentiality Probe
    lambda_path = Path("data/benchmarks/lambda_essentiality.csv")
    lambda_metrics = {"acc": 0.0, "f1": 0.0, "mcc": 0.0}
    if lambda_path.exists():
        print(f"[*] Extracting embeddings for Lambda Phage essentiality...")
        df_lambda = pd.read_csv(lambda_path)
        X_lambda = extract_embeddings_for_seqs(
            model, df_lambda["sequence"].tolist(), stoi, bos_id, eos_id, pad_id, block_size, device
        )
        y_lambda = df_lambda["essential"].to_numpy()
        lambda_metrics = run_linear_probe(X_lambda, y_lambda)
        print(f"  - Lambda Phage Essentiality (5-fold CV):")
        print(f"    Accuracy: {lambda_metrics['acc']:.4f}")
        print(f"    F1 Score: {lambda_metrics['f1']:.4f}")
        print(f"    MCC:      {lambda_metrics['mcc']:.4f}")
    else:
        print(f"[!] Lambda essentiality dataset not found at {lambda_path}")

    # 3. Pseudomonas aeruginosa Gene Essentiality Probe
    pa_path = Path("data/benchmarks/pseudomonas_essentiality.csv")
    pa_metrics = {"acc": 0.0, "f1": 0.0, "mcc": 0.0}
    if pa_path.exists():
        print(f"[*] Extracting embeddings for Pseudomonas aeruginosa essentiality...")
        df_pa = pd.read_csv(pa_path)
        X_pa = extract_embeddings_for_seqs(
            model, df_pa["sequence"].tolist(), stoi, bos_id, eos_id, pad_id, block_size, device
        )
        y_pa = df_pa["essential"].to_numpy()
        pa_metrics = run_linear_probe(X_pa, y_pa)
        print(f"  - Pseudomonas aeruginosa Essentiality (5-fold CV):")
        print(f"    Accuracy: {pa_metrics['acc']:.4f}")
        print(f"    F1 Score: {pa_metrics['f1']:.4f}")
        print(f"    MCC:      {pa_metrics['mcc']:.4f}")
    else:
        print(f"[!] Pseudomonas essentiality dataset not found at {pa_path}")

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
        "sota_lambda_essentiality_acc": lambda_metrics["acc"],
        "sota_lambda_essentiality_f1": lambda_metrics["f1"],
        "sota_lambda_essentiality_mcc": lambda_metrics["mcc"],
        "sota_pseudomonas_essentiality_acc": pa_metrics["acc"],
        "sota_pseudomonas_essentiality_f1": pa_metrics["f1"],
        "sota_pseudomonas_essentiality_mcc": pa_metrics["mcc"]
    })
    
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"[*] Wrote gene essentiality linear probing results to {metrics_path}")

if __name__ == "__main__":
    main()
