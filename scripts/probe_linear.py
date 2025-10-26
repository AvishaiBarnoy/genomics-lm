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
import torch
import torch.nn.functional as F

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


def _train_probe(train_x: torch.Tensor, train_y: torch.Tensor, num_classes: int) -> torch.nn.Module:
    model = torch.nn.Linear(train_x.shape[1], num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    for _ in range(EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_x)
        loss = F.cross_entropy(logits, train_y)
        loss.backward()
        optimizer.step()
    return model


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

    torch.manual_seed(RNG_SEED)

    token_to_idx = stoi(tokens)
    indices = [token_to_idx[row["token"]] for row in rows if row.get("token") in token_to_idx]
    X = torch.tensor(embeddings[indices], dtype=torch.float32)

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
        y_task = torch.tensor(y[valid_idx], dtype=torch.long)

        fold_acc = []
        for train_idx, val_idx in _kfold_indices(X_task.shape[0], K_FOLDS):
            train_x = X_task[train_idx]
            val_x = X_task[val_idx]
            train_y = y_task[train_idx]
            val_y = y_task[val_idx]
            if train_x.shape[0] == 0 or val_x.shape[0] == 0:
                continue
            num_classes = int(y_task.max().item() + 1)
            model = _train_probe(train_x, train_y, num_classes)
            with torch.no_grad():
                preds = model(val_x).argmax(dim=1)
                acc = (preds == val_y).float().mean().item()
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
