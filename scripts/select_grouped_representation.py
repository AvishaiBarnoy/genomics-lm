#!/usr/bin/env python3
"""Select a pooled representation using grouped CV without test-set access."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from src.classifiers.linear_probe import fit_logreg
from src.classifiers.probes import compute_metrics
from src.codonlm.evaluation_provenance import bind_embedding_artifact


def _mapping(path: Path, value_column: str) -> dict[str, str]:
    with path.open(newline="") as handle:
        return {
            row["id"]: row[value_column]
            for row in csv.DictReader(
                handle, delimiter="\t" if path.suffix == ".tsv" else ","
            )
            if row.get("id") and row.get(value_column)
        }


def _load(path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=True) as blob:
        ids = [str(value) for value in blob["ids"]]
        arrays = {
            key.removeprefix("X__"): np.asarray(blob[key])
            for key in blob.files
            if key.startswith("X__")
        }
    if not arrays:
        raise ValueError(f"no multi-representation arrays found in {path}")
    return ids, arrays


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--group-column", default="protein_cluster")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--primary-metric", default="macro_auprc")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    labels = _mapping(args.labels, "label")
    groups = _mapping(args.groups, args.group_column)
    loaded = []
    for path in args.embeddings:
        bind_embedding_artifact(path, require_verified=True)
        loaded.append((path, *_load(path)))
    reference_ids = loaded[0][1]
    candidates = set(loaded[0][2])
    for path, ids, arrays in loaded[1:]:
        if ids != reference_ids:
            raise ValueError(f"embedding ID order differs: {path}")
        candidates &= set(arrays)
    keep = [
        index
        for index, identifier in enumerate(reference_ids)
        if identifier in labels and identifier in groups
    ]
    ids = [reference_ids[index] for index in keep]
    label_values = sorted({labels[identifier] for identifier in ids})
    label_to_int = {label: index for index, label in enumerate(label_values)}
    y = np.asarray([label_to_int[labels[identifier]] for identifier in ids])
    group_values = np.asarray([groups[identifier] for identifier in ids])
    splitter = StratifiedGroupKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed
    )
    splits = list(splitter.split(np.zeros(len(ids)), y, group_values))

    reports = []
    for candidate in sorted(candidates):
        per_run = []
        all_folds = []
        for path, _, arrays in loaded:
            X = arrays[candidate][keep]
            fold_metrics = []
            for train_index, val_index in splits:
                result = fit_logreg(X[train_index], y[train_index], C=args.C)
                prediction = result.model.predict(X[val_index])
                probability = result.model.predict_proba(X[val_index])
                metrics = compute_metrics(y[val_index], prediction, probability)
                fold_metrics.append(metrics)
                all_folds.append(metrics)
            per_run.append(
                {
                    "embedding": str(path),
                    "folds": fold_metrics,
                }
            )
        keys = sorted(set.intersection(*(set(fold) for fold in all_folds)))
        aggregate = {
            key: {
                "mean": float(np.mean([fold[key] for fold in all_folds])),
                "std": float(np.std([fold[key] for fold in all_folds])),
            }
            for key in keys
        }
        reports.append(
            {
                "representation": candidate,
                "aggregate": aggregate,
                "runs": per_run,
            }
        )

    reports.sort(
        key=lambda item: item["aggregate"].get(
            args.primary_metric, {"mean": float("-inf")}
        )["mean"],
        reverse=True,
    )
    output = {
        "protocol": "stratified_protein_cluster_grouped_cv",
        "selection_split": "amr_train_only",
        "folds": args.folds,
        "seed": args.seed,
        "C": args.C,
        "primary_metric": args.primary_metric,
        "records": len(ids),
        "groups": len(set(group_values)),
        "classes": label_values,
        "winner": reports[0]["representation"],
        "ranking": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(
        f"[representation] winner={output['winner']} "
        f"{args.primary_metric}={reports[0]['aggregate'][args.primary_metric]['mean']:.4f}"
    )


if __name__ == "__main__":
    main()
