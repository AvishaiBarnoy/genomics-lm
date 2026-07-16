#!/usr/bin/env python3
"""
scripts/audit_duplicates.py — Pretraining Split Cross-Leakage Auditor

This script audits pretraining splits for exact sequence duplicates and near-duplicates
using sliding-window codon subsequence sharing (for window sizes L = 10, 20, and 30 codons).

Usage:
  python -m scripts.audit_duplicates --train_npz data/processed/train_bs256.npz --test_npz data/processed/test_bs256.npz
"""

import argparse
from pathlib import Path
import numpy as np

def extract_sequence_tokens(seq_ids):
    # Strip padding (value 0)
    return [int(val) for val in seq_ids if val != 0]

def get_sliding_lmers(seq, L):
    if len(seq) < L:
        return set()
    return {tuple(seq[i : i + L]) for i in range(len(seq) - L + 1)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_npz", help="Path to train NPZ dataset")
    ap.add_argument("--test_npz", help="Path to test NPZ dataset")
    args = ap.parse_args()

    # 1. Resolve Paths
    train_path = args.train_npz
    test_path = args.test_npz

    if not train_path or not test_path:
        # Fallback to defaults
        base_dir = Path("data/processed")
        if not train_path:
            paths = list(base_dir.glob("**/train*.npz"))
            train_path = paths[0] if paths else base_dir / "train_bs256.npz"
        if not test_path:
            paths = list(base_dir.glob("**/test*.npz"))
            test_path = paths[0] if paths else base_dir / "test_bs256.npz"

    train_path = Path(train_path)
    test_path = Path(test_path)

    if not train_path.exists():
        raise FileNotFoundError(f"Train NPZ not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test NPZ not found: {test_path}")

    print(f"[audit] Loading train dataset: {train_path}...")
    with np.load(train_path) as data:
        X_train = data["X"]

    print(f"[audit] Loading test dataset: {test_path}...")
    with np.load(test_path) as data:
        X_test = data["X"]

    print(f"[audit] Extracted {len(X_train)} training sequences, {len(X_test)} test sequences.")

    # 2. Extract clean sense codon token sequences
    train_seqs = [extract_sequence_tokens(seq) for seq in X_train]
    test_seqs = [extract_sequence_tokens(seq) for seq in X_test]

    # 3. Exact Duplicate Check
    print("[audit] Running exact sequence duplicate check...")
    train_hashes = {tuple(seq) for seq in train_seqs if seq}
    exact_duplicates = 0
    for seq in test_seqs:
        if not seq:
            continue
        if tuple(seq) in train_hashes:
            exact_duplicates += 1

    exact_pct = (exact_duplicates / len(test_seqs)) * 100 if test_seqs else 0.0
    print(f"  Exact duplicate sequences in test split: {exact_duplicates} / {len(test_seqs)} ({exact_pct:.2f}%)")

    # 4. Near-Duplicate Check (Contiguous Subsequence Sharing)
    print("[audit] Running sliding-window near-duplicate check...")
    window_sizes = [10, 20, 30]
    
    print("\n| Window Size (Codons / bp) | Unique Train L-mers | Shared Test Sequences | Leakage Percentage |")
    print("| :--- | :---: | :---: | :---: |")

    for L in window_sizes:
        bp = L * 3
        # Index all unique L-mers in the training set
        train_lmers = set()
        for seq in train_seqs:
            train_lmers.update(get_sliding_lmers(seq, L))

        # Check test sequences for overlap
        shared_seqs = 0
        for seq in test_seqs:
            test_lmers = get_sliding_lmers(seq, L)
            if not test_lmers.isdisjoint(train_lmers):
                shared_seqs += 1

        leak_pct = (shared_seqs / len(test_seqs)) * 100 if test_seqs else 0.0
        print(f"| {L:2} codons ({bp:2} bp)       | {len(train_lmers):19,} | {shared_seqs:21,} | {leak_pct:17.2f}% |")

if __name__ == "__main__":
    main()
