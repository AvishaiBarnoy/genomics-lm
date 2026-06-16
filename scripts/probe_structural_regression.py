#!/usr/bin/env python3
"""
Evaluates structural awareness of genomics-lm hidden states using supervised regression probes.
Fits Ridge Regression models mapping hidden states to physical DNAshape parameters
using 5-fold cross-validation, and saves the trained models to skip future training.
"""

import argparse
import json
import os
import random
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from scripts._shared import load_model, resolve_run, load_token_list
from scripts.probe_structural_awareness import get_theoretical_shape
from src.codonlm.metrics_io import write_merge_metrics

def dev() -> torch.device:
    import os
    if os.environ.get("FORCE_CPU") == "1":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

class NPZ:
    def __init__(self, path):
        self.data = np.load(path, allow_pickle=False)
        self.is_dynamic = "lengths" in self.data
        if self.is_dynamic:
            self.seqs = np.split(self.data["X"], np.cumsum(self.data["lengths"])[:-1])

    def __len__(self):
        return len(self.seqs) if self.is_dynamic else len(self.data["X"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id")
    ap.add_argument("--run_dir")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--test_npz", default="data/processed/stage2.5_master_pack_v2/test_bs512.npz")
    ap.add_argument("--sample_size", type=int, default=150, help="Number of sequences to extract representations from")
    ap.add_argument("--no_cache", action="store_true", help="Do not load pre-trained probes from disk")
    args = ap.parse_args()

    run_id, run_dir = resolve_run(run_id=args.run_id, run_dir=args.run_dir)
    device = dev()
    print(f"[*] Probing run: {run_id} using device: {device}")

    # Resolve paths
    checkpoint_dir = run_dir / "checkpoints"
    scores_dir = run_dir / "scores"
    probe_model_path = checkpoint_dir / "structural_regression_probes.pt"

    model, spec = load_model(run_dir, device=device, ckpt_name=args.ckpt)
    tokens = load_token_list(run_dir)
    stoi = {t: i for i, t in enumerate(tokens)}
    itos = tokens

    # Load test dataset sequences
    if not os.path.exists(args.test_npz):
        raise FileNotFoundError(f"Test dataset not found at {args.test_npz}")
    ds = NPZ(args.test_npz)

    # 1. Check if we have pre-trained probes saved
    regression_probes = {}
    if not args.no_cache and probe_model_path.exists():
        print(f"[+] Found saved structural regression probes at {probe_model_path}. Loading...")
        regression_probes = torch.load(probe_model_path, map_location="cpu", weights_only=False)

    # Pick a stable set of test sequences for evaluation
    random.seed(42)
    n_samples = min(args.sample_size, len(ds))
    idxs = list(range(len(ds)))
    random.shuffle(idxs)
    selected_idxs = idxs[:n_samples]

    # 2. Extract Hidden States and compute DNAshape targets
    print(f"[*] Extracting representations and DNAshape targets from {n_samples} test sequences...")
    X_list, Y_dict = [], {
        "MGW": [], "Roll": [], "EP": [], "ProT": [], "HelT": [],
        "Slide": [], "Rise": [], "Shift": [], "Tilt": [],
        "Buckle": [], "Opening": [], "Shear": [], "Stagger": [], "Stretch": []
    }

    for idx in selected_idxs:
        seq_tokens = ds.seqs[idx].tolist() if ds.is_dynamic else ds.data["X"][idx].tolist()
        # Filter padding/separators out
        codons = [itos[tid] for tid in seq_tokens if tid > 0 and tid < len(itos)]
        if len(codons) < 10:
            continue

        # Reconstruct DNA string
        dna_seq = "".join(codons)

        # Get targets
        targets = get_theoretical_shape(dna_seq)

        # Extract model hidden states
        input_ids = torch.tensor([[stoi[c] for c in codons if c in stoi]], device=device).long()
        with torch.no_grad():
            x = model.tok_emb(input_ids) + model.pos_emb(
                torch.arange(input_ids.size(1), device=device).unsqueeze(0)
            )
            for block in model.blocks:
                x = block(x)
            hidden_states = x.squeeze(0).cpu().numpy()  # (T, D)

        # Pool targets per codon
        pooled_targets = {}
        for name, values in targets.items():
            codon_values = []
            for i in range(0, len(values) - 2, 3):
                codon_values.append(values[i : i + 3].mean())
            pooled_targets[name] = np.array(codon_values[:len(hidden_states)])

        X_list.append(hidden_states)
        for name in Y_dict:
            Y_dict[name].append(pooled_targets[name])

    # Concatenate all steps into large regression matrices
    X = np.concatenate(X_list, axis=0) # (Total_Codons, D)
    print(f"[*] Combined regression matrix shape: X={X.shape}")

    metrics = {}

    # 3. Train or Evaluate
    for name in Y_dict:
        Y = np.concatenate(Y_dict[name], axis=0) # (Total_Codons,)

        if name in regression_probes:
            # We already have a saved probe for this property, evaluate it directly
            probe = regression_probes[name]
            preds = probe.predict(X)

            # Compute R^2 and Pearson correlation
            mean_y = np.mean(Y)
            ss_tot = np.sum((Y - mean_y) ** 2)
            ss_res = np.sum((Y - preds) ** 2)
            r2 = 1.0 - (ss_res / max(1e-9, ss_tot))

            corr = float(np.corrcoef(preds, Y)[0, 1]) if np.std(preds) > 0 and np.std(Y) > 0 else 0.0
            print(f"[Cached Probe] {name} -> R^2: {r2:.4f} | Pearson: {corr:.4f}")

            metrics[f"regression_probe.{name}.r2"] = r2
            metrics[f"regression_probe.{name}.corr"] = corr
        else:
            # Fit Ridge Regression using 5-fold cross validation to assess generalization
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            fold_r2, fold_corr = [], []

            for train_idx, test_idx in kf.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                Y_train, Y_test = Y[train_idx], Y[test_idx]

                clf = Ridge(alpha=1.0)
                clf.fit(X_train, Y_train)
                preds = clf.predict(X_test)

                # Compute fold metrics
                r2 = clf.score(X_test, Y_test)
                corr = float(np.corrcoef(preds, Y_test)[0, 1]) if np.std(preds) > 0 and np.std(Y_test) > 0 else 0.0
                fold_r2.append(r2)
                fold_corr.append(corr)

            avg_r2 = float(np.mean(fold_r2))
            avg_corr = float(np.mean(fold_corr))
            print(f"[Training Probe] {name} -> 5-Fold Cross-Val R^2: {avg_r2:.4f} | Pearson: {avg_corr:.4f}")

            metrics[f"regression_probe.{name}.r2"] = avg_r2
            metrics[f"regression_probe.{name}.corr"] = avg_corr

            # Fit final model on all data and cache it
            final_clf = Ridge(alpha=1.0)
            final_clf.fit(X, Y)
            regression_probes[name] = final_clf

    # Save newly trained probes
    print(f"[*] Saving trained structural regression probes to {probe_model_path}...")
    torch.save(regression_probes, probe_model_path)

    # Compute overall averages
    avg_r2 = np.mean([metrics[f"regression_probe.{k}.r2"] for k in Y_dict])
    avg_corr = np.mean([metrics[f"regression_probe.{k}.corr"] for k in Y_dict])
    metrics["regression_probe.avg_r2"] = float(avg_r2)
    metrics["regression_probe.avg_corr"] = float(avg_corr)
    print("------------------------------------------")
    print(f"Overall Regression Probe R^2 Score: {avg_r2:.4f} | Avg Pearson: {avg_corr:.4f}")

    # Merge into metrics.json
    write_merge_metrics(scores_dir / "metrics.json", metrics)
    print(f"[+] Wrote regression probe metrics to {scores_dir / 'metrics.json'}")

if __name__ == "__main__":
    main()
