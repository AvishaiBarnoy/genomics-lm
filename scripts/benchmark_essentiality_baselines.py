#!/usr/bin/env python3
"""
Gene essentiality benchmark comparing:
1. Logistic Regression on raw codon frequencies (local linear baseline)
2. Gradient Boosting (HistGradientBoostingClassifier) on raw codon frequencies (non-linear baseline)
3. Logistic Regression on pre-trained CodonLM embeddings (representation probe)
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

from scripts._shared import resolve_run, load_model
from scripts.extract_embeddings import _pool_hidden, _dna_to_codon_tokens

def extract_codon_freqs(seqs, stoi):
    vocab_size = len(stoi)
    features = []
    for seq in seqs:
        # Split sequence into 3-bp codons
        codons = [seq[i:i+3] for i in range(0, len(seq) - 2, 3)]
        counts = np.zeros(vocab_size, dtype=np.float32)
        total = 0
        for c in codons:
            if c in stoi:
                counts[stoi[c]] += 1.0
                total += 1
        if total > 0:
            counts /= total
        features.append(counts)
    return np.stack(features, axis=0)

def extract_lm_embeddings(model, seqs, stoi, bos_id, eos_id, pad_id, block_size, device):
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
                embeddings.append(np.zeros(model.tok_emb.embedding_dim))
                continue

            ids_tensor = torch.tensor(
                toks[:block_size], dtype=torch.long, device=device
            ).unsqueeze(0)
            nonpad = ids_tensor.ne(pad_id)
            pooled = _pool_hidden(model, ids_tensor, nonpad)
            embeddings.append(pooled.squeeze(0).cpu().numpy())

    return np.stack(embeddings, axis=0)

def evaluate_model(X, y, classifier_type):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, f1s, mccs = [], [], []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if classifier_type == "lr":
            clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        elif classifier_type == "gbdt":
            clf = HistGradientBoostingClassifier(random_state=42)

        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        accs.append(accuracy_score(y_test, preds))
        f1s.append(f1_score(y_test, preds, average='binary', zero_division=0))
        mccs.append(matthews_corrcoef(y_test, preds))

    return {
        "acc": np.mean(accs),
        "f1": np.mean(f1s),
        "mcc": np.mean(mccs)
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    args = ap.parse_args()

    run_id, run_dir = resolve_run(run_id=args.run_id)
    device = torch.device("cpu") # Enforce CPU to avoid MPS conflicts

    model, spec = load_model(run_dir, device=device, ckpt_name=args.ckpt)
    from scripts import query_model as Q
    itos, stoi = Q._load_vocab(run_dir)

    bos_id = stoi.get("<BOS_CDS>")
    eos_id = stoi.get("<EOS_CDS>")
    pad_id = stoi.get("<PAD>", 0)
    block_size = spec.block_size if hasattr(spec, 'block_size') else 512

    datasets = {
        "Lambda Phage Essentiality": Path("data/benchmarks/lambda_essentiality.csv"),
        "Pseudomonas aeruginosa": Path("data/benchmarks/pseudomonas_essentiality.csv")
    }

    print("=========================================================================================")
    print(f"Gene Essentiality Baselines comparison for: {run_id} ({args.ckpt})")
    print("=========================================================================================")

    for name, path in datasets.items():
        if not path.exists():
            print(f"[!] Dataset {name} not found at {path}. Skipping.")
            continue

        print(f"\n[*] Evaluating on: {name}")
        df = pd.read_csv(path)
        seqs = df["sequence"].tolist()
        y = df["essential"].to_numpy()

        # Extract features
        X_freqs = extract_codon_freqs(seqs, stoi)
        X_embeddings = extract_lm_embeddings(model, seqs, stoi, bos_id, eos_id, pad_id, block_size, device)

        # 1. Codon Frequencies + Logistic Regression
        res_freq_lr = evaluate_model(X_freqs, y, "lr")
        # 2. Codon Frequencies + GBDT
        res_freq_gbdt = evaluate_model(X_freqs, y, "gbdt")
        # 3. LM Embeddings + Logistic Regression
        res_emb_lr = evaluate_model(X_embeddings, y, "lr")

        print("-----------------------------------------------------------------------------------------")
        print(f"{'Metric':<12} | {'Codon Freqs Linear':<20} | {'Codon Freqs GBDT':<20} | {'LM Embeddings Linear':<20}")
        print("-----------------------------------------------------------------------------------------")
        for metric in ["acc", "f1", "mcc"]:
            print(f"{metric.upper():<12} | {res_freq_lr[metric]:.4f}{'':<14} | {res_freq_gbdt[metric]:.4f}{'':<14} | {res_emb_lr[metric]:.4f}")
        print("-----------------------------------------------------------------------------------------")

if __name__ == "__main__":
    main()
