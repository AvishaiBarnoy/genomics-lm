#!/usr/bin/env python3
"""
Baseline benchmark script comparing:
1. Ridge regression on one-hot sequence context (local linear baseline)
2. Gradient Boosting (HistGradientBoosting) on one-hot sequence context (non-linear baseline)
3. Ridge regression on CodonLM's pre-trained hidden states (representation probe)
"""

import argparse
import random
import os
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from scripts._shared import load_model, resolve_run, load_token_list
from scripts.probe_structural_awareness import get_theoretical_shape

class NPZ:
    def __init__(self, path):
        self.data = np.load(path, allow_pickle=False)
        self.is_dynamic = "lengths" in self.data
        if self.is_dynamic:
            self.seqs = np.split(self.data["X"], np.cumsum(self.data["lengths"])[:-1])

    def __len__(self):
        return len(self.seqs) if self.is_dynamic else len(self.data["X"])

def one_hot_context(seq_tokens, vocab_size, window_half=1):
    # For each position t, extract a one-hot context of size 2*window_half + 1
    T = len(seq_tokens)
    features = np.zeros((T, (2 * window_half + 1) * vocab_size), dtype=np.float32)
    for t in range(T):
        feat_idx = 0
        for offset in range(-window_half, window_half + 1):
            neighbor_t = t + offset
            if 0 <= neighbor_t < T:
                token_id = seq_tokens[neighbor_t]
                if 0 <= token_id < vocab_size:
                    features[t, feat_idx * vocab_size + token_id] = 1.0
            feat_idx += 1
    return features

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--test_npz", default="data/processed/stage2.5_master_pack_v2/test_bs512.npz")
    ap.add_argument("--sample_size", type=int, default=100, help="Number of sequences to evaluate")
    args = ap.parse_args()

    run_id, run_dir = resolve_run(run_id=args.run_id)
    device = torch.device("cpu") # Enforce CPU

    # Load model and token mapping
    model, spec = load_model(run_dir, device=device, ckpt_name=args.ckpt)
    tokens = load_token_list(run_dir)
    stoi = {t: i for i, t in enumerate(tokens)}
    itos = tokens
    vocab_size = len(itos)

    # Load test dataset
    if not os.path.exists(args.test_npz):
        raise FileNotFoundError(f"Test dataset not found at {args.test_npz}")
    ds = NPZ(args.test_npz)

    random.seed(42)
    n_samples = min(args.sample_size, len(ds))
    idxs = list(range(len(ds)))
    random.shuffle(idxs)
    selected_idxs = idxs[:n_samples]

    print(f"[*] Extracting features from {n_samples} test sequences...")
    X_onehot_list = []
    X_hidden_list = []

    Y_dict = {
        "MGW": [], "Roll": [], "EP": [], "ProT": [], "HelT": []
    }

    for idx in selected_idxs:
        seq_tokens = ds.seqs[idx].tolist() if ds.is_dynamic else ds.data["X"][idx].tolist()
        codons = [itos[tid] for tid in seq_tokens if tid > 0 and tid < len(itos)]
        if len(codons) < 10:
            continue

        # 1. One-hot features (sliding window of context)
        sub_tokens = [stoi[c] for c in codons if c in stoi]
        onehot_feats = one_hot_context(sub_tokens, vocab_size=vocab_size, window_half=1)

        # 2. Extract hidden states
        input_ids = torch.tensor([sub_tokens], device=device).long()
        with torch.no_grad():
            x = model.tok_emb(input_ids) + model.pos_emb(torch.arange(input_ids.size(1), device=device).unsqueeze(0))
            for block in model.blocks:
                x = block(x)
            hidden_states = x.squeeze(0).cpu().numpy()

        # 3. DNAshape Targets
        dna_seq = "".join(codons)
        targets = get_theoretical_shape(dna_seq)

        # Pool targets per codon
        pooled_targets = {}
        for name in Y_dict:
            values = targets[name]
            codon_values = []
            for i in range(0, len(values) - 2, 3):
                codon_values.append(values[i : i + 3].mean())
            pooled_targets[name] = np.array(codon_values[:len(hidden_states)])

        X_onehot_list.append(onehot_feats)
        X_hidden_list.append(hidden_states)
        for name in Y_dict:
            Y_dict[name].append(pooled_targets[name])

    X_onehot = np.concatenate(X_onehot_list, axis=0)
    X_hidden = np.concatenate(X_hidden_list, axis=0)

    print(f"[*] One-hot matrix shape: {X_onehot.shape}")
    print(f"[*] Hidden state matrix shape: {X_hidden.shape}")
    print("---------------------------------------------------------------------------------")
    print(f"{'DNAshape Property':<20} | {'One-hot Linear':<18} | {'One-hot GBDT (GB)':<18} | {'LM Linear (Ridge)':<18}")
    print(f"{'':<20} | {'R2 / Corr':<18} | {'R2 / Corr':<18} | {'R2 / Corr':<18}")
    print("---------------------------------------------------------------------------------")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for name in Y_dict:
        Y = np.concatenate(Y_dict[name], axis=0)

        # 1. Local One-hot Linear (Ridge)
        r2_oh_lin, corr_oh_lin = [], []
        for train, test in kf.split(X_onehot):
            clf = Ridge(alpha=1.0)
            clf.fit(X_onehot[train], Y[train])
            preds = clf.predict(X_onehot[test])
            r2_oh_lin.append(clf.score(X_onehot[test], Y[test]))
            corr_oh_lin.append(float(np.corrcoef(preds, Y[test])[0, 1]) if np.std(preds) > 0 else 0.0)

        # 2. Local One-hot GBDT (HistGradientBoosting)
        r2_oh_gbdt, corr_oh_gbdt = [], []
        for train, test in kf.split(X_onehot):
            clf = HistGradientBoostingRegressor(max_iter=50, random_state=42)
            # Use smaller sample of training fold for speed
            sample_train = train[:1500] if len(train) > 1500 else train
            clf.fit(X_onehot[sample_train], Y[sample_train])
            preds = clf.predict(X_onehot[test])
            r2_oh_gbdt.append(clf.score(X_onehot[test], Y[test]))
            corr_oh_gbdt.append(float(np.corrcoef(preds, Y[test])[0, 1]) if np.std(preds) > 0 else 0.0)

        # 3. LM Hidden State Linear (Ridge)
        r2_lm, corr_lm = [], []
        for train, test in kf.split(X_hidden):
            clf = Ridge(alpha=1.0)
            clf.fit(X_hidden[train], Y[train])
            preds = clf.predict(X_hidden[test])
            r2_lm.append(clf.score(X_hidden[test], Y[test]))
            corr_lm.append(float(np.corrcoef(preds, Y[test])[0, 1]) if np.std(preds) > 0 else 0.0)

        # Print results for this property
        oh_lin_str = f"{np.mean(r2_oh_lin):.3f} / {np.mean(corr_oh_lin):.3f}"
        oh_gb_str = f"{np.mean(r2_oh_gbdt):.3f} / {np.mean(corr_oh_gbdt):.3f}"
        lm_str = f"{np.mean(r2_lm):.3f} / {np.mean(corr_lm):.3f}"
        print(f"{name:<20} | {oh_lin_str:<18} | {oh_gb_str:<18} | {lm_str:<18}")

    print("---------------------------------------------------------------------------------")

if __name__ == "__main__":
    main()
