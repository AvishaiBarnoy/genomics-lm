"""Linear and MLP probes on token embeddings.

If probe_labels.csv is missing, attempt to generate it from the run's
`itos.txt` using the standard genetic code mapping.
"""

from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from scipy.stats import pearsonr

from ._shared import (
    ensure_run_layout,
    load_artifacts,
    load_token_list,
    stoi,
    resolve_run,
)

try:
    from .generate_probe_labels import (
        STANDARD_GENETIC_CODE,
        POLARITY_CLASS,
        HYDROPATHY_CLASS,
        START_CODONS,
        classify_codon,
    )
except Exception:
    STANDARD_GENETIC_CODE = {}
    POLARITY_CLASS = {}
    HYDROPATHY_CLASS = {}
    START_CODONS = set()
    classify_codon = None


def _write_probe_labels_if_missing(run_dir: Path, tokens: list[str]) -> None:
    path = run_dir / "probe_labels.csv"
    if path.exists():
        return
    if classify_codon is None:
        return
    rows = []
    for tok in tokens:
        codon = tok.upper()
        if len(codon) == 3 and codon.isalpha():
            aa, polarity, hyd, is_stop, is_start, kd, mw, pi = classify_codon(codon)
        else:
            aa = polarity = hyd = is_stop = is_start = kd = mw = pi = ""
        rows.append((tok, aa, polarity, hyd, is_stop, is_start, kd, mw, pi))
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["token", "aa", "polarity", "hydropathy", "is_stop", "is_start", "kd_hydropathy", "mw_volume", "pi"]
        )
        for r in rows:
            writer.writerow(list(r))


CLASSIFICATION_TASKS = {
    "AA identity": "aa",
    "polarity class": "polarity",
    "hydropathy class": "hydropathy",
    "is_stop": "is_stop",
    "is_start": "is_start",
}

REGRESSION_TASKS = {
    "Hydropathy Index (KD)": "kd_hydropathy",
    "Molecular Weight (MW)": "mw_volume",
    "Isoelectric Point (pI)": "pi",
}

K_FOLDS = 5
RNG_SEED = 1337


def _read_probe_labels(path: Path, tokens: List[str]) -> List[Dict[str, str]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            token = row.get("token")
            if token is None or token not in tokens:
                continue
            rows.append({k: v for k, v in row.items()})
    return rows


def _encode_labels(
    rows: List[Dict[str, str]], field: str
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    valid = [r[field] for r in rows if r.get(field) not in (None, "")]
    unique = sorted(set(valid))
    mapping = {v: i for i, v in enumerate(unique)}
    labels = []
    mask = []
    for r in rows:
        val = r.get(field)
        if val in (None, ""):
            mask.append(False)
            labels.append(0)
        else:
            mask.append(True)
            labels.append(mapping[val])
    return np.array(labels, dtype=np.int64), np.array(mask, dtype=bool), mapping


def _encode_binary(
    rows: List[Dict[str, str]], field: str
) -> Tuple[np.ndarray, np.ndarray]:
    def to_int(val: str) -> int:
        if val is None:
            return -1
        val = val.strip().lower()
        if val in {"1", "true", "yes", "y"}:
            return 1
        if val in {"0", "false", "no", "n"}:
            return 0
        return -1

    labels = []
    mask = []
    for r in rows:
        value = to_int(r.get(field, ""))
        if value < 0:
            mask.append(False)
            labels.append(0)
        else:
            mask.append(True)
            labels.append(value)
    return np.array(labels, dtype=np.int64), np.array(mask, dtype=bool)


def _encode_continuous(
    rows: List[Dict[str, str]], field: str
) -> Tuple[np.ndarray, np.ndarray]:
    labels = []
    mask = []
    for r in rows:
        val = r.get(field)
        if val in (None, ""):
            mask.append(False)
            labels.append(0.0)
        else:
            try:
                labels.append(float(val))
                mask.append(True)
            except ValueError:
                mask.append(False)
                labels.append(0.0)
    return np.array(labels, dtype=np.float32), np.array(mask, dtype=bool)


def _train_eval_linear_sklearn(
    train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray
) -> float:
    clf = LogisticRegression(max_iter=200)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        clf.fit(train_x, train_y)
        return float(clf.score(val_x, val_y))


def _train_eval_mlp_sklearn(
    train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray
) -> float:
    clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=RNG_SEED)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        clf.fit(train_x, train_y)
        return float(clf.score(val_x, val_y))


def _train_eval_regression(
    train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray
) -> Tuple[float, float]:
    reg = Ridge(alpha=1.0)
    reg.fit(train_x, train_y)
    preds = reg.predict(val_x)
    
    r2 = float(reg.score(val_x, val_y))
    
    if len(preds) >= 2 and np.std(preds) > 1e-9 and np.std(val_y) > 1e-9:
        corr, _ = pearsonr(preds, val_y)
        pearson = float(corr)
    else:
        pearson = 0.0
    return r2, pearson


def main(argv: Optional[Iterable[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id", nargs="?")
    ap.add_argument("--run_dir", help="Alternative to run_id; path to runs/<RUN_ID>")
    args = ap.parse_args(argv)

    run_id, run_dir = resolve_run(args.run_id, args.run_dir)
    paths = ensure_run_layout(run_id)
    run_dir, tables_dir = paths["run"], paths["tables"]

    embeddings = load_artifacts(run_id).get("token_embeddings")
    if embeddings is None or embeddings.size == 0:
        print("[probe-linear] token embeddings missing; aborting")
        return
    tokens = load_token_list(run_dir)

    label_path = run_dir / "probe_labels.csv"
    _write_probe_labels_if_missing(run_dir, tokens)
    rows = _read_probe_labels(label_path, tokens)
    if not rows:
        print(f"[probe-linear] probe_labels.csv missing or empty at {label_path}")
        return

    token_to_idx = stoi(tokens)
    indices = [
        token_to_idx[row["token"]] for row in rows if row.get("token") in token_to_idx
    ]
    X = embeddings[indices].astype(np.float32, copy=False)

    results = []

    # 1. Classification Tasks (Linear & MLP)
    for task, field in CLASSIFICATION_TASKS.items():
        if field in {"is_stop", "is_start"}:
            y, mask = _encode_binary(rows, field)
        else:
            y, mask, _ = _encode_labels(rows, field)
        if mask.sum() < K_FOLDS:
            print(f"[probe-linear] skipping classification task {task}; insufficient labels")
            continue
        valid_idx = np.where(mask)[0]
        X_task = X[valid_idx]
        y_task = y[valid_idx].astype(np.int64, copy=False)

        # Drop singleton classes for AA identity
        classes, counts = np.unique(y_task, return_counts=True)
        if field == "aa":
            keep_classes = set(int(c) for c, cnt in zip(classes, counts) if cnt >= 2)
            if keep_classes and len(keep_classes) < classes.size:
                keep_mask = np.array(
                    [int(v) in keep_classes for v in y_task], dtype=bool
                )
                X_task = X_task[keep_mask]
                y_task = y_task[keep_mask]
                classes, counts = np.unique(y_task, return_counts=True)
                print(f"[probe-linear] AA identity: dropped singleton classes; kept {classes.size} classes")

        if classes.size < 2:
            print(f"[probe-linear] skipping classification task {task}; only one class present")
            continue

        min_per_class = int(counts.min())
        
        # 1a. Linear & MLP Probes
        if min_per_class < 2 or len(y_task) < K_FOLDS:
            test_size = 0.5 if min_per_class == 1 else 0.2
            try:
                train_x, val_x, train_y, val_y = train_test_split(
                    X_task, y_task, test_size=test_size, stratify=y_task, random_state=RNG_SEED
                )
                linear_accs = [_train_eval_linear_sklearn(train_x, train_y, val_x, val_y)]
                mlp_accs = [_train_eval_mlp_sklearn(train_x, train_y, val_x, val_y)]
            except ValueError:
                continue
        else:
            n_splits = max(2, min(K_FOLDS, min_per_class))
            linear_accs = []
            mlp_accs = []
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RNG_SEED)
            for train_idx, val_idx in skf.split(X_task, y_task):
                train_x, val_x = X_task[train_idx], X_task[val_idx]
                train_y, val_y = y_task[train_idx], y_task[val_idx]
                if np.unique(train_y).size >= 2 and np.unique(val_y).size >= 2:
                    linear_accs.append(_train_eval_linear_sklearn(train_x, train_y, val_x, val_y))
                    mlp_accs.append(_train_eval_mlp_sklearn(train_x, train_y, val_x, val_y))

        if linear_accs:
            results.append((task, "linear_class", "accuracy", float(np.mean(linear_accs)), float(np.std(linear_accs))))
        if mlp_accs:
            results.append((task, "mlp_class", "accuracy", float(np.mean(mlp_accs)), float(np.std(mlp_accs))))

    # 2. Continuous Regression Tasks (Ridge)
    for task, field in REGRESSION_TASKS.items():
        y, mask = _encode_continuous(rows, field)
        if mask.sum() < K_FOLDS:
            print(f"[probe-linear] skipping regression task {task}; insufficient labels")
            continue
        valid_idx = np.where(mask)[0]
        X_task = X[valid_idx]
        y_task = y[valid_idx]

        r2_scores = []
        pearson_scores = []
        kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RNG_SEED)
        for train_idx, val_idx in kf.split(X_task, y_task):
            train_x, val_x = X_task[train_idx], X_task[val_idx]
            train_y, val_y = y_task[train_idx], y_task[val_idx]
            r2, p_corr = _train_eval_regression(train_x, train_y, val_x, val_y)
            r2_scores.append(r2)
            pearson_scores.append(p_corr)

        if r2_scores:
            results.append((task, "ridge_reg", "r2", float(np.mean(r2_scores)), float(np.std(r2_scores))))
            results.append((task, "ridge_reg", "pearson", float(np.mean(pearson_scores)), float(np.std(pearson_scores))))

    out_path = tables_dir / "probe_results.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "probe_type", "metric_name", "mean_score", "std_score"])
        for task, p_type, metric, mean, std in results:
            writer.writerow([task, p_type, metric, f"{mean:.4f}", f"{std:.4f}"])

    print(f"[probe-linear] wrote results to {out_path}")


if __name__ == "__main__":
    main()
