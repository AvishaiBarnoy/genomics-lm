"""Linear probes on token embeddings.

If probe_labels.csv is missing, attempt to generate it from the run's
`itos.txt` using the standard genetic code mapping.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split

from ._shared import ensure_run_layout, load_artifacts, load_token_list, stoi

try:
    # Reuse mappings from the label generator when available
    from .generate_probe_labels import (
        STANDARD_GENETIC_CODE,
        POLARITY_CLASS,
        HYDROPATHY_CLASS,
        START_CODONS,
    )
except Exception:
    STANDARD_GENETIC_CODE = {}
    POLARITY_CLASS = {}
    HYDROPATHY_CLASS = {}
    START_CODONS = set()


def _write_probe_labels_if_missing(run_dir: Path, tokens: list[str]) -> None:
    path = run_dir / "probe_labels.csv"
    if path.exists():
        return
    if not STANDARD_GENETIC_CODE:
        # generator not available; do nothing
        return
    rows = []
    stop_codons = {c for c, aa in STANDARD_GENETIC_CODE.items() if aa == "Stop"}
    for tok in tokens:
        codon = tok.upper()
        aa = polarity = hyd = ""
        is_stop = is_start = "0"
        if len(codon) == 3 and codon.isalpha():
            aa_or_stop = STANDARD_GENETIC_CODE.get(codon)
            if aa_or_stop == "Stop":
                aa = "Stop"
                is_stop = "1"
            elif aa_or_stop is not None:
                aa = aa_or_stop
                polarity = POLARITY_CLASS.get(aa, "")
                hyd = HYDROPATHY_CLASS.get(aa, "")
                if codon in stop_codons:
                    is_stop = "1"
                if codon in START_CODONS:
                    is_start = "1"
        rows.append((tok, aa, polarity, hyd, is_stop, is_start))
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["token", "aa", "polarity", "hydropathy", "is_stop", "is_start"])
        for r in rows:
            writer.writerow(list(r))

TASKS = {
    "AA identity": "aa",
    "polarity class": "polarity",
    "hydropathy class": "hydropathy",
    "is_stop": "is_stop",
    "is_start": "is_start",
}
K_FOLDS = 5
EPOCHS = 300
LR = 0.1
WEIGHT_DECAY = 1e-4
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


def _encode_labels(rows: List[Dict[str, str]], field: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
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


def _encode_binary(rows: List[Dict[str, str]], field: str) -> Tuple[np.ndarray, np.ndarray]:
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


def _kfold_indices(n: int, k: int, seed: int = RNG_SEED):
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    fold_sizes = [n // k + (1 if i < n % k else 0) for i in range(k)]
    start = 0
    for size in fold_sizes:
        end = start + size
        val_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])
        yield train_idx, val_idx
        start = end


def _train_eval_probe_sklearn(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray) -> float:
    # Keep defaults; sklearn>=1.5 deprecates multi_class="auto", so omit it
    clf = LogisticRegression(max_iter=200)
    clf.fit(train_x, train_y)
    return float(clf.score(val_x, val_y))


def main(argv: Optional[Iterable[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    args = ap.parse_args(argv)

    paths = ensure_run_layout(args.run_id)
    run_dir, tables_dir = paths["run"], paths["tables"]

    embeddings = load_artifacts(args.run_id).get("token_embeddings")
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
    indices = [token_to_idx[row["token"]] for row in rows if row.get("token") in token_to_idx]
    X = embeddings[indices].astype(np.float32, copy=False)

    results = []
    for task, field in TASKS.items():
        if field in {"is_stop", "is_start"}:
            y, mask = _encode_binary(rows, field)
        else:
            y, mask, _ = _encode_labels(rows, field)
        if mask.sum() < K_FOLDS:
            print(f"[probe-linear] skipping {task}; insufficient labels")
            continue
        valid_idx = np.where(mask)[0]
        X_task = X[valid_idx]
        y_task = y[valid_idx].astype(np.int64, copy=False)

        # Require at least 2 classes overall
        classes, counts = np.unique(y_task, return_counts=True)
        if classes.size < 2:
            print(f"[probe-linear] skipping {task}; only one class present")
            continue
        min_per_class = int(counts.min())
        # If extremely sparse, fall back to holdout split (stratified)
        if min_per_class < 2 or len(y_task) < K_FOLDS:
            test_size = 0.5 if min_per_class == 1 else 0.2
            train_x, val_x, train_y, val_y = train_test_split(
                X_task, y_task, test_size=test_size, stratify=y_task, random_state=RNG_SEED
            )
            if np.unique(train_y).size < 2 or np.unique(val_y).size < 2:
                print(f"[probe-linear] skipping {task}; holdout had <2 classes")
                continue
            acc = _train_eval_probe_sklearn(train_x, train_y, val_x, val_y)
            fold_acc = [acc]
        else:
            n_splits = max(2, min(K_FOLDS, min_per_class))
            fold_acc = []
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RNG_SEED)
            for train_idx, val_idx in skf.split(X_task, y_task):
                train_x = X_task[train_idx]
                val_x = X_task[val_idx]
                train_y = y_task[train_idx]
                val_y = y_task[val_idx]
                # Ensure both splits have at least 2 classes
                if train_x.shape[0] == 0 or val_x.shape[0] == 0:
                    continue
                if np.unique(train_y).size < 2 or np.unique(val_y).size < 2:
                    continue
                acc = _train_eval_probe_sklearn(train_x, train_y, val_x, val_y)
                fold_acc.append(acc)
        if not fold_acc:
            continue
        results.append((task, float(np.mean(fold_acc)), float(np.std(fold_acc))))

    out_path = tables_dir / "probe_results.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "mean_accuracy", "std_accuracy"])
        for task, mean, std in results:
            writer.writerow([task, f"{mean:.4f}", f"{std:.4f}"])

    print(f"[probe-linear] wrote results to {out_path}")


if __name__ == "__main__":
    main()
