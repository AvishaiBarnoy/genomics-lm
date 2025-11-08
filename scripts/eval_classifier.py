#!/usr/bin/env python3
"""Evaluate a saved classifier on provided embeddings or sequences.

CLI examples:
  # Embeddings
  python -m scripts.eval_classifier --kind probe --model outputs/reports/e1/model.pkl \
    --embeddings outputs/reports/e1/test_embeddings.npz --labels data/labels/test.csv --out outputs/reports/e1

  # K-mer
  python -m scripts.eval_classifier --kind kmer --model outputs/reports/e1/model.pkl \
    --vectorizer outputs/reports/e1/vectorizer.pkl --seqs data/test_seqs.csv --labels data/labels/test.csv --out outputs/reports/e1
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import json
import joblib
import numpy as np

from src.classifiers.probes import ensure_dir, load_npz, save_json, compute_metrics, plot_confusion, plot_calibration


def _read_labels_csv(path: Path) -> dict:
    lab = {}
    with path.open("r", newline="") as f:
        for row in csv.DictReader(f):
            lab[row["id"]] = int(row["label"]) if row["label"].isdigit() else row["label"]
    return lab


def _align_embeddings(npz_path: Path, labels_path: Path):
    data = load_npz(npz_path)
    X = np.asarray(data["X"])  # (N, D)
    ids = [str(x) for x in data.get("ids", [str(i) for i in range(X.shape[0])])]
    lab_map = _read_labels_csv(labels_path)
    y = []
    keep = []
    for i, sid in enumerate(ids):
        if sid in lab_map:
            y.append(lab_map[sid]); keep.append(i)
    X2 = X[np.array(keep)]
    uniq = sorted(set(y))
    map_to_int = {v: i for i, v in enumerate(uniq)}
    y2 = np.array([map_to_int[v] for v in y], dtype=np.int64)
    return X2, y2


def _read_seqs_csv(path: Path):
    ids, seqs = [], []
    with path.open("r", newline="") as f:
        for row in csv.DictReader(f):
            ids.append(row["id"]); seqs.append(row["seq"])
    return ids, seqs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["probe", "kmer", "mlp"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--vectorizer")
    ap.add_argument("--embeddings")
    ap.add_argument("--seqs")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_dir = ensure_dir(args.out)
    if args.kind in {"probe", "mlp"}:
        X, y = _align_embeddings(Path(args.embeddings), Path(args.labels))
        if args.kind == "mlp":
            import torch
            from src.classifiers.mlp_head import MLP
            state = torch.load(args.model, map_location="cpu")
            n_classes = int(np.max(y)) + 1
            d_in = X.shape[1]
            model = MLP(d_in, n_classes)
            model.load_state_dict(state); model.eval()
            with torch.no_grad():
                logits = model(torch.from_numpy(X.astype(np.float32)))
                y_pred = torch.argmax(logits, dim=1).cpu().numpy()
                y_proba = torch.softmax(logits, dim=1).cpu().numpy()
        else:
            model = joblib.load(args.model)
            y_pred = model.predict(X)
            y_proba = None
            try:
                y_proba = model.predict_proba(X)
            except Exception:
                try:
                    y_proba = model.decision_function(X)
                except Exception:
                    pass
    else:
        model = joblib.load(args.model)
        vec = joblib.load(args.vectorizer)
        _, seqs = _read_seqs_csv(Path(args.seqs))
        X = vec.transform(seqs)
        y_map = _read_labels_csv(Path(args.labels))
        ids, _ = _read_seqs_csv(Path(args.seqs))
        y = np.array([y_map[i] for i in ids], dtype=object)
        uniq = sorted(set(y)); map_to_int = {v: i for i, v in enumerate(uniq)}
        y = np.array([map_to_int[v] for v in y], dtype=np.int64)
        y_pred = model.predict(X)
        y_proba = None
        try:
            y_proba = model.predict_proba(X)
        except Exception:
            try:
                y_proba = model.decision_function(X)
            except Exception:
                pass

    metrics = compute_metrics(y, y_pred, y_proba)
    save_json(out_dir / "metrics_eval.json", metrics)
    plot_confusion(y, y_pred, out_dir / "confusion_eval.png")
    if y_proba is not None:
        plot_calibration(y, y_proba, out_dir / "calibration_eval.png")
    print(f"[eval-clf] wrote reports to {out_dir}")


if __name__ == "__main__":
    main()

