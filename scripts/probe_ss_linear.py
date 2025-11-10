#!/usr/bin/env python3
"""
Token-level linear probe for secondary structure (H/E/C) given labels.

Inputs:
  --emb_npz path  NPZ with arrays:
      H: (N, T, D) token embeddings (e.g., final hidden states)
      Y: (N, T)    integer labels in {0:C, 1:H, 2:E}
      M: (N, T)    optional mask (1=valid token, 0=pad)
  --out_dir path  where to write metrics.json and confusion.png

Outputs:
  - metrics.json: accuracy, macro_f1, AUROC (OVR when possible)
  - confusion.png

Notes:
  - This script does NOT generate labels; provide labels from an external SS tool or dataset.
  - If H is (N,T,D) and too large, consider subsampling or pooling before running.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import json


def load_npz(path: Path):
    with np.load(path, allow_pickle=False) as blob:
        return {k: blob[k] for k in blob.files}


def plot_confusion(y_true, y_pred, out_path: Path):
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(4,4))
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_title("SS confusion (normalized)")
    plt.tight_layout(); out_path.parent.mkdir(parents=True, exist_ok=True); plt.savefig(out_path); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_npz", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    data = load_npz(Path(args.emb_npz))
    H = data["H"]  # (N, T, D)
    Y = data["Y"]  # (N, T)
    M = data.get("M")

    if M is None:
        M = np.ones_like(Y, dtype=np.int64)

    # Flatten masked tokens
    mask = (M.reshape(-1) > 0)
    X = H.reshape(-1, H.shape[-1])[mask]
    y = Y.reshape(-1)[mask]

    clf = Pipeline([
        ("scaler", StandardScaler(with_mean=True)),
        ("clf", LogisticRegression(max_iter=2000, n_jobs=-1, multi_class="ovr"))
    ])
    clf.fit(X, y)
    y_pred = clf.predict(X)
    metrics = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "macro_f1": float(f1_score(y, y_pred, average="macro")),
    }
    try:
        y_score = clf.decision_function(X)
        # one-vs-rest AUROC
        # build one-hot
        K = int(np.max(y)) + 1
        Y_oh = np.zeros((y.shape[0], K), dtype=np.float32)
        Y_oh[np.arange(y.shape[0]), y] = 1.0
        metrics["auroc_ovr"] = float(roc_auc_score(Y_oh, y_score, multi_class="ovr"))
    except Exception:
        pass

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_confusion(y, y_pred, out_dir / "confusion.png")
    joblib.dump(clf, out_dir / "probe.pkl")
    print(f"[probe-ss] wrote metrics and plots to {out_dir}")


if __name__ == "__main__":
    main()

