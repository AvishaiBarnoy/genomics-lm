#!/usr/bin/env python3
"""Train classifiers on LM embeddings or k-mer baselines.

Config YAML keys (examples in configs/classifier):
  protocol: std|TSTR|TRTS
  task: name-for-outputs
  out_dir: outputs/reports/<task>

  data:
    # Embedding paths and labels (CSV with columns: id,label)
    train_embeddings: path/to/train_embeddings.npz
    train_labels: path/to/train_labels.csv
    test_embeddings: path/to/test_embeddings.npz
    test_labels: path/to/test_labels.csv

    # For baselines, sequences instead of embeddings
    train_seqs: path/to/train_sequences.csv   # columns: id,seq
    test_seqs: path/to/test_sequences.csv

  classifier:
    kind: probe_logreg | probe_svm | mlp | kmer_logreg | kmer_svm | kmer_xgb
    # probe params
    C: 1.0
    # mlp params
    hidden: 128
    depth: 1
    epochs: 20
    lr: 0.001
    batch_size: 64
    # kmer params
    k: 3
    tfidf: true

Outputs:
  - metrics.json with accuracy/macro-F1/AUROC
  - confusion.png and calibration.png
  - model.pkl (sklearn) or model.pt (MLP) and vectorizer.pkl for k-mer
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np

from src.classifiers.probes import ensure_dir, load_yaml, save_json, load_npz, compute_metrics, plot_confusion, plot_calibration
from src.classifiers.linear_probe import fit_linear_svm, fit_logreg
from src.classifiers.mlp_head import fit_mlp
from src.classifiers.kmer_baselines import fit_kmer_logreg, fit_kmer_svm, fit_kmer_xgb


def _read_labels_csv(path: Path) -> Dict[str, int]:
    lab = {}
    with path.open("r", newline="") as f:
        for row in csv.DictReader(f):
            lab[row["id"]] = int(row["label"]) if row["label"].isdigit() else row["label"]
    return lab


def _align_embeddings(npz_path: Path, labels_path: Path) -> Tuple[np.ndarray, np.ndarray]:
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
    # normalize labels to integer classes if strings
    uniq = sorted(set(y))
    map_to_int = {v: i for i, v in enumerate(uniq)}
    y2 = np.array([map_to_int[v] for v in y], dtype=np.int64)
    return X2, y2


def _read_seqs_csv(path: Path) -> Tuple[list, list]:
    ids, seqs = [], []
    with path.open("r", newline="") as f:
        for row in csv.DictReader(f):
            ids.append(row["id"]); seqs.append(row["seq"])
    return ids, seqs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    out_dir = ensure_dir(cfg.get("out_dir", "outputs/reports/exp"))
    proto = str(cfg.get("protocol", "std")).upper()
    clf_kind = str(cfg.get("classifier", {}).get("kind", "probe_logreg"))

    if clf_kind.startswith("probe") or clf_kind == "mlp":
        train_X, train_y = _align_embeddings(Path(cfg["data"]["train_embeddings"]), Path(cfg["data"]["train_labels"]))
        test_X, test_y = _align_embeddings(Path(cfg["data"]["test_embeddings"]), Path(cfg["data"]["test_labels"]))
        if clf_kind == "probe_logreg":
            res = fit_logreg(train_X, train_y, C=float(cfg["classifier"].get("C", 1.0)))
            model = res.model; y_pred = model.predict(test_X)
            y_proba = None
            try:
                y_proba = model.predict_proba(test_X)
            except Exception:
                pass
        elif clf_kind == "probe_svm":
            res = fit_linear_svm(train_X, train_y, C=float(cfg["classifier"].get("C", 1.0)))
            model = res.model; y_pred = model.predict(test_X)
            y_proba = None
            try:
                y_proba = model.decision_function(test_X)
            except Exception:
                pass
        else:  # mlp
            res = fit_mlp(
                train_X, train_y,
                epochs=int(cfg["classifier"].get("epochs", 20)),
                lr=float(cfg["classifier"].get("lr", 1e-3)),
                batch_size=int(cfg["classifier"].get("batch_size", 64)),
                hidden=int(cfg["classifier"].get("hidden", 128)),
                depth=int(cfg["classifier"].get("depth", 1)),
                dropout=float(cfg["classifier"].get("dropout", 0.1)),
            )
            model = res.model
            import torch
            with torch.no_grad():
                logits = model(torch.from_numpy(test_X.astype(np.float32)))
                y_pred = torch.argmax(logits, dim=1).cpu().numpy()
                y_proba = torch.softmax(logits, dim=1).cpu().numpy()
        metrics = compute_metrics(test_y, y_pred, y_proba)
        save_json(out_dir / "metrics.json", metrics)
        plot_confusion(test_y, y_pred, out_dir / "confusion.png")
        if y_proba is not None:
            plot_calibration(test_y, y_proba, out_dir / "calibration.png")
        # save model
        if clf_kind == "mlp":
            import torch
            torch.save(model.state_dict(), out_dir / "model.pt")
        else:
            joblib.dump(model, out_dir / "model.pkl")
        print(f"[train-clf] {clf_kind} → {out_dir}; metrics={json.dumps(metrics)}")

    else:
        # k-mer baselines expect sequences
        _, train_seqs = _read_seqs_csv(Path(cfg["data"]["train_seqs"]))
        _, test_seqs = _read_seqs_csv(Path(cfg["data"]["test_seqs"]))
        # Align labels by id order in CSVs
        train_lab = _read_labels_csv(Path(cfg["data"]["train_labels"]))
        test_lab = _read_labels_csv(Path(cfg["data"]["test_labels"]))
        train_ids, _ = _read_seqs_csv(Path(cfg["data"]["train_seqs"]))
        test_ids, _ = _read_seqs_csv(Path(cfg["data"]["test_seqs"]))
        y_train = np.array([train_lab[i] for i in train_ids], dtype=object)
        y_test = np.array([test_lab[i] for i in test_ids], dtype=object)
        # map labels to ints
        uniq = sorted(set(y_train) | set(y_test))
        mapping = {v: i for i, v in enumerate(uniq)}
        y_train_i = np.array([mapping[v] for v in y_train], dtype=np.int64)
        y_test_i = np.array([mapping[v] for v in y_test], dtype=np.int64)

        k = int(cfg["classifier"].get("k", 3)); tfidf = bool(cfg["classifier"].get("tfidf", True))
        if clf_kind == "kmer_logreg":
            res = fit_kmer_logreg(train_seqs, y_train_i, k=k, tfidf=tfidf)
            vectorizer, model = res.vectorizer, res.model
            Xte = vectorizer.transform(test_seqs)
            y_pred = model.predict(Xte)
            try:
                y_proba = model.predict_proba(Xte)
            except Exception:
                y_proba = None
        elif clf_kind == "kmer_svm":
            res = fit_kmer_svm(train_seqs, y_train_i, k=k, tfidf=tfidf)
            vectorizer, model = res.vectorizer, res.model
            Xte = vectorizer.transform(test_seqs)
            y_pred = model.predict(Xte)
            try:
                y_proba = model.decision_function(Xte)
            except Exception:
                y_proba = None
        else:
            res = fit_kmer_xgb(train_seqs, y_train_i, k=k, tfidf=tfidf)
            vectorizer, model = res.vectorizer, res.model
            Xte = vectorizer.transform(test_seqs)
            y_pred = model.predict(Xte)
            try:
                y_proba = model.predict_proba(Xte)
            except Exception:
                y_proba = None
        metrics = compute_metrics(y_test_i, y_pred, y_proba)
        save_json(out_dir / "metrics.json", metrics)
        plot_confusion(y_test_i, y_pred, out_dir / "confusion.png")
        if y_proba is not None:
            plot_calibration(y_test_i, y_proba, out_dir / "calibration.png")
        joblib.dump(model, out_dir / "model.pkl"); joblib.dump(vectorizer, out_dir / "vectorizer.pkl")
        print(f"[train-clf] {clf_kind} → {out_dir}; metrics={json.dumps(metrics)}")


if __name__ == "__main__":
    main()

