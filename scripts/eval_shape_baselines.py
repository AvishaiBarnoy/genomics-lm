#!/usr/bin/env python3
"""
scripts/eval_shape_baselines.py — DNA-Shape Probing Control Suite

This script evaluates and compares Ridge regression prediction performance (R^2 scores)
using three representation spaces:
  1. Pretrained Model Embeddings (Genomics-LM)
  2. Randomly Initialized Model Embeddings (structural baseline)
  3. Raw One-Hot Codon Representations (sequence-identity baseline)

It runs 5-fold cross-validation over sense codons in the test set to determine
if pretraining provides a statistically significant improvement.
"""

import argparse
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

from scripts._shared import load_model, build_model, resolve_run, load_token_list, stoi
from scripts.probe_structural_awareness import get_theoretical_shape

def extract_hidden_states(model, input_ids):
    with torch.no_grad():
        if hasattr(model, "forward_hidden"):
            # Exposes causal & segment-masked states
            h = model.forward_hidden(input_ids)
        else:
            x = model.tok_emb(input_ids)
            if not getattr(model, "use_rope", False):
                x = x + model.pos_emb(torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0))
            x = model.drop(x)
            for block in model.blocks:
                x = block(x)
            h = x
        return h.squeeze(0).cpu().numpy()  # (T, D)

def evaluate_features_r2(X, Y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        
        reg = Ridge(alpha=1.0)
        reg.fit(X_train, Y_train)
        preds = reg.predict(X_test)
        scores.append(r2_score(Y_test, preds))
    return np.mean(scores)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id", nargs="?")
    ap.add_argument("--run_dir", help="Path to runs/<RUN_ID>")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--test_npz", help="Path to test NPZ dataset")
    ap.add_argument("--max_seqs", type=int, default=50, help="Max sequences to process")
    args = ap.parse_args()

    run_id, run_dir = resolve_run(args.run_id, args.run_dir)
    tokens = load_token_list(run_dir)
    itos = {i: t for i, t in enumerate(tokens)}
    vocab_size = len(tokens)

    # 1. Load Pretrained and Random Models
    print(f"[baselines] Loading pretrained model from {run_dir}...")
    pretrained_model, spec = load_model(run_dir, ckpt_name=args.ckpt)
    
    print("[baselines] Initializing random model with identical architecture...")
    random_model = build_model(spec)

    # 2. Locate Test Set
    test_path = args.test_npz
    if not test_path:
        # Try to find test NPZ inside the global manifest run dir or processed dir
        global_test = run_dir / "test_bs256.npz"
        if global_test.exists():
            test_path = global_test
        else:
            # Fallback to standard processed path
            test_path = Path("data/processed/test_bs256.npz")
            if not test_path.exists():
                # Search for any test NPZ in data/processed
                paths = list(Path("data/processed").glob("**/test*.npz"))
                if paths:
                    test_path = paths[0]
                else:
                    raise FileNotFoundError("Could not find test NPZ file. Please specify --test_npz.")

    test_path = Path(test_path)
    print(f"[baselines] Loading test sequences from {test_path}...")
    with np.load(test_path) as data:
        X_test = data["X"]

    # Limit sequence count for performance
    n_seqs = min(len(X_test), args.max_seqs)
    X_test = X_test[:n_seqs]
    print(f"[baselines] Processing {n_seqs} test sequences...")

    # 3. Extract Representations and DNA-Shape Targets
    all_pretrained = []
    all_random = []
    all_one_hot = []
    
    # Store lists of target values for all 14 properties
    properties = [
        "MGW", "Roll", "EP", "ProT", "HelT",
        "Slide", "Rise", "Shift", "Tilt",
        "Buckle", "Opening", "Shear", "Stagger", "Stretch"
    ]
    all_targets = {p: [] for p in properties}

    for idx in range(n_seqs):
        seq_ids = X_test[idx]
        
        # Filter out special tokens (0: PAD, 1: BOS, 2: EOS, 3: SEP)
        sense_indices = [i for i, val in enumerate(seq_ids) if val >= 4]
        if not sense_indices:
            continue
            
        sense_ids = seq_ids[sense_indices]
        
        # Build contiguous DNA sequence from sense codons
        dna_seq = "".join(itos[val] for val in sense_ids)
        
        # Calculate theoretical shape targets (base-pair level)
        shape_targets = get_theoretical_shape(dna_seq)
        
        # Pool targets per codon (average over 3 bases)
        pooled_targets = {}
        for prop_name, values in shape_targets.items():
            codon_values = []
            for i in range(0, len(values) - 2, 3):
                codon_values.append(values[i : i + 3].mean())
            pooled_targets[prop_name] = np.array(codon_values[:len(sense_ids)])

        # Run forward pass on both models for this sequence
        input_tensor = torch.tensor([seq_ids]).long()
        h_pretrained = extract_hidden_states(pretrained_model, input_tensor)[sense_indices]
        h_random = extract_hidden_states(random_model, input_tensor)[sense_indices]

        # One-hot representation of sense codons
        one_hot = np.zeros((len(sense_ids), vocab_size), dtype=np.float32)
        one_hot[np.arange(len(sense_ids)), sense_ids] = 1.0

        all_pretrained.append(h_pretrained)
        all_random.append(h_random)
        all_one_hot.append(one_hot)

        for prop_name in properties:
            all_targets[prop_name].append(pooled_targets[prop_name])

    # Stack all codon representations
    X_pretrained = np.vstack(all_pretrained)
    X_random = np.vstack(all_random)
    X_one_hot = np.vstack(all_one_hot)

    for prop_name in properties:
        all_targets[prop_name] = np.concatenate(all_targets[prop_name])

    print(f"[baselines] Features extracted for {X_pretrained.shape[0]} sense codons.")
    print(f"  Pretrained feature dim: {X_pretrained.shape[1]}")
    print(f"  One-hot feature dim:    {X_one_hot.shape[1]}")

    # 4. Fit Ridge Regression and Compare R^2
    print("\n| DNA-Shape Property | One-Hot R^2 | Random Model R^2 | Pretrained Model R^2 | Delta (Pretrained - One-Hot) |")
    print("| :--- | :---: | :---: | :---: | :---: |")
    
    r2_pretrained_list = []
    r2_onehot_list = []

    for prop_name in properties:
        Y = all_targets[prop_name]
        
        r2_one_hot = evaluate_features_r2(X_one_hot, Y)
        r2_random = evaluate_features_r2(X_random, Y)
        r2_pretrained = evaluate_features_r2(X_pretrained, Y)
        
        delta = r2_pretrained - r2_one_hot
        print(f"| {prop_name:18s} | {r2_one_hot:10.4f} | {r2_random:16.4f} | {r2_pretrained:20.4f} | {delta:28.4f} |")
        
        r2_pretrained_list.append(r2_pretrained)
        r2_onehot_list.append(r2_one_hot)

    mean_pre = np.mean(r2_pretrained_list)
    mean_oh = np.mean(r2_onehot_list)
    print(f"| {'Mean':18s} | {mean_oh:10.4f} | {'N/A':16s} | {mean_pre:20.4f} | {mean_pre - mean_oh:28.4f} |")

if __name__ == "__main__":
    main()
