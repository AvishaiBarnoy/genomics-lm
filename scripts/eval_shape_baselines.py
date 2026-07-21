#!/usr/bin/env python3
"""Leakage-controlled DNA-shape representation controls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from scripts._shared import build_model, load_model, load_token_list, resolve_run
from scripts.probe_structural_awareness import get_theoretical_shape
from src.codonlm.checkpoints import load_codon_checkpoint
from src.codonlm.evaluation_provenance import (
    bind_checkpoint_dataset,
    bind_dataset_manifest,
)

PROPERTIES = (
    "MGW", "Roll", "EP", "ProT", "HelT", "Slide", "Rise", "Shift", "Tilt",
    "Buckle", "Opening", "Shear", "Stagger", "Stretch",
)
METHODS = ("one_hot", "local_5mer", "local_7mer", "random", "pretrained")


def extract_hidden_states(model, input_ids):
    forward_hidden = getattr(model, "forward_hidden", None)
    if not callable(forward_hidden):
        raise TypeError(f"{type(model).__name__} lacks the verified forward_hidden API")
    if getattr(model, "use_shape_guidance", False):
        raise RuntimeError("shape-guided models require the dedicated artifact-aware extractor")
    with torch.no_grad():
        return forward_hidden(input_ids).squeeze(0).cpu().numpy()


def make_group_folds(groups: np.ndarray, n_splits: int, seed: int):
    unique, counts = np.unique(groups, return_counts=True)
    if len(unique) < n_splits:
        raise ValueError(f"need at least {n_splits} groups, found {len(unique)}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))
    order = order[np.argsort(-counts[order], kind="stable")]
    fold_sizes = np.zeros(n_splits, dtype=np.int64)
    assignment = {}
    for group_index in order:
        fold = int(np.argmin(fold_sizes))
        assignment[str(unique[group_index])] = fold
        fold_sizes[fold] += counts[group_index]
    folds = []
    for fold in range(n_splits):
        test = np.array([assignment[str(group)] == fold for group in groups])
        train_idx, test_idx = np.flatnonzero(~test), np.flatnonzero(test)
        if not len(train_idx) or not len(test_idx):
            raise ValueError(f"fold {fold} is empty")
        folds.append((train_idx, test_idx))
    return folds, assignment


def _local_mer(dna: str, codon_index: int, size: int) -> str:
    center = codon_index * 3 + 1
    radius = size // 2
    padded = "N" * radius + dna + "N" * radius
    center += radius
    return padded[center - radius : center + radius + 1]


def _load_windows(path: Path):
    with np.load(path, allow_pickle=False) as data:
        x = np.asarray(data["X"])
        if "lengths" not in data:
            return [row for row in x]
        lengths = np.asarray(data["lengths"])
    offsets = np.concatenate([[0], np.cumsum(lengths[:-1])])
    return [x[int(start) : int(start + length)] for start, length in zip(offsets, lengths)]


def _read_spans(path: Path):
    by_window = defaultdict(list)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            by_window[int(row["window_index"])].append(row)
    return by_window


def _read_genomes(path: Path | None):
    if path is None:
        return {}
    with path.open(newline="") as handle:
        return {
            row["source_id"]: row["genome"]
            for row in csv.DictReader(handle, delimiter="\t")
        }


def collect_features(
    windows, spans, genomes, tokens, pretrained, random_model, group_by, max_windows
):
    pretrained_rows, random_rows, token_rows = [], [], []
    mer5_rows, mer7_rows, groups, sample_ids = [], [], [], []
    targets = {name: [] for name in PROPERTIES}
    for window_index, sequence in enumerate(windows[:max_windows]):
        input_ids = torch.as_tensor(sequence, dtype=torch.long).unsqueeze(0)
        hidden_pre = extract_hidden_states(pretrained, input_ids)
        hidden_random = extract_hidden_states(random_model, input_ids)
        for span_index, span in enumerate(spans.get(window_index, [])):
            start, end = int(span["window_token_start"]), int(span["window_token_end"])
            positions = [pos for pos in range(start, min(end, len(sequence))) if int(sequence[pos]) >= 4]
            if not positions:
                continue
            codons = [tokens[int(sequence[pos])] for pos in positions]
            if any(len(codon) != 3 or set(codon) - set("ACGT") for codon in codons):
                raise ValueError(f"non-canonical sense token in window {window_index}, span {span_index}")
            dna = "".join(codons)
            shape = get_theoretical_shape(dna)
            source = span["source_id"]
            if group_by == "genome":
                if source not in genomes:
                    raise ValueError(f"missing genome for source_id={source}")
                group = genomes[source]
            elif group_by == "gene":
                group = source
            else:
                group = f"window:{window_index}"
            for codon_index, position in enumerate(positions):
                pretrained_rows.append(hidden_pre[position])
                random_rows.append(hidden_random[position])
                token_rows.append(int(sequence[position]))
                mer5_rows.append({_local_mer(dna, codon_index, 5): 1.0})
                mer7_rows.append({_local_mer(dna, codon_index, 7): 1.0})
                groups.append(group)
                sample_ids.append(f"{window_index}:{source}:{position}")
                for name in PROPERTIES:
                    values = shape[name][codon_index * 3 : codon_index * 3 + 3]
                    targets[name].append(float(np.mean(values)))
    if not groups:
        raise ValueError("no evaluable codon positions")
    one_hot = np.zeros((len(token_rows), len(tokens)), dtype=np.float32)
    one_hot[np.arange(len(token_rows)), token_rows] = 1.0
    vectorizer5, vectorizer7 = DictVectorizer(sparse=True), DictVectorizer(sparse=True)
    features = {
        "one_hot": one_hot,
        "local_5mer": vectorizer5.fit_transform(mer5_rows),
        "local_7mer": vectorizer7.fit_transform(mer7_rows),
        "random": np.asarray(random_rows),
        "pretrained": np.asarray(pretrained_rows),
    }
    return features, {key: np.asarray(value) for key, value in targets.items()}, np.asarray(groups), sample_ids


def _summary(values):
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("non-finite fold score; increase held-out group/sample counts")
    mean = float(np.mean(values))
    if len(values) < 2:
        return {"mean": mean, "ci95": [mean, mean]}
    margin = float(stats.t.ppf(0.975, len(values) - 1) * stats.sem(values))
    return {"mean": mean, "ci95": [mean - margin, mean + margin]}


def evaluate(features, targets, folds):
    results = {name: {} for name in METHODS}
    for method in METHODS:
        for prop, y in targets.items():
            scores = []
            for train_idx, test_idx in folds:
                model = Ridge(alpha=1.0)
                model.fit(features[method][train_idx], y[train_idx])
                scores.append(float(r2_score(y[test_idx], model.predict(features[method][test_idx]))))
            results[method][prop] = {"fold_scores": scores, **_summary(scores)}
    aggregate = {}
    for method in METHODS:
        fold_scores = [
            float(np.mean([results[method][prop]["fold_scores"][fold] for prop in PROPERTIES]))
            for fold in range(len(folds))
        ]
        aggregate[method] = {"fold_scores": fold_scores, **_summary(fold_scores)}
    paired = {}
    pretrained = np.asarray(aggregate["pretrained"]["fold_scores"])
    for baseline in METHODS[:-1]:
        differences = pretrained - np.asarray(aggregate[baseline]["fold_scores"])
        comparison = _summary(differences)
        pvalue = float(stats.ttest_rel(pretrained, aggregate[baseline]["fold_scores"]).pvalue)
        comparison["pvalue_paired_t"] = pvalue if np.isfinite(pvalue) else None
        paired[baseline] = comparison
    return results, aggregate, paired


def _sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id", nargs="?")
    parser.add_argument("--run_dir")
    parser.add_argument("--ckpt", default="best.pt")
    parser.add_argument("--test_npz", required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--packing-metadata", required=True, type=Path)
    parser.add_argument("--cds-metadata", type=Path)
    parser.add_argument("--group-by", choices=("window", "gene", "genome"), default="gene")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-seqs", "--max_seqs", dest="max_seqs", type=int, default=50)
    parser.add_argument("--output-prefix", required=True, type=Path)
    args = parser.parse_args()
    if args.group_by == "genome" and args.cds_metadata is None:
        parser.error("--cds-metadata is required for --group-by genome")
    run_id, run_dir = resolve_run(args.run_id, args.run_dir)
    manifest_provenance = {"status": "legacy_unverified"}
    if args.manifest is not None:
        expected = {
            "test_tokens": Path(args.test_npz),
            "test_packing_metadata": args.packing_metadata,
        }
        if args.cds_metadata is not None:
            expected["source_metadata"] = args.cds_metadata
        _, manifest_provenance = bind_dataset_manifest(
            args.manifest, expected_artifacts=expected, require_scientific=False
        )
    _, checkpoint_cfg, _ = load_codon_checkpoint(run_dir, ckpt_name=args.ckpt)
    checkpoint_dataset = bind_checkpoint_dataset(
        checkpoint_cfg,
        manifest_provenance if args.manifest is not None else None,
    )
    tokens = load_token_list(run_dir)
    pretrained, spec = load_model(run_dir, ckpt_name=args.ckpt)
    random_model = build_model(spec).eval()
    checkpoint_path = run_dir / "checkpoints" / args.ckpt
    if not checkpoint_path.exists():
        checkpoint_path = run_dir / args.ckpt
    vocabulary_path = run_dir / "itos.txt"
    test_path = Path(args.test_npz)
    features, targets, groups, sample_ids = collect_features(
        _load_windows(test_path), _read_spans(args.packing_metadata),
        _read_genomes(args.cds_metadata), tokens, pretrained, random_model,
        args.group_by, args.max_seqs,
    )
    folds, assignments = make_group_folds(groups, args.n_splits, args.seed)
    results, aggregate, paired = evaluate(features, targets, folds)
    report = {
        "schema_version": 1, "run_id": run_id, "seed": args.seed,
        "dataset_manifest": manifest_provenance,
        "checkpoint_dataset": checkpoint_dataset,
        "group_by": args.group_by, "n_splits": args.n_splits,
        "dataset": {"path": str(test_path.resolve()), "sha256": _sha256(test_path)},
        "checkpoint": {"path": str(checkpoint_path.resolve()), "sha256": _sha256(checkpoint_path)},
        "vocabulary": {"path": str(vocabulary_path.resolve()), "sha256": _sha256(vocabulary_path), "size": len(tokens)},
        "packing_metadata": {"path": str(args.packing_metadata.resolve()), "sha256": _sha256(args.packing_metadata)},
        "cds_metadata": ({"path": str(args.cds_metadata.resolve()), "sha256": _sha256(args.cds_metadata)} if args.cds_metadata else None),
        "n_positions": len(groups), "group_assignments": assignments,
        "feature_context": {"local_5mer": "centered", "local_7mer": "centered"},
        "ridge_alpha": 1.0,
        "results": results, "aggregate": aggregate, "paired_vs_pretrained": paired,
    }
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    with args.output_prefix.with_suffix(".folds.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["sample_id", "group_id", "fold"])
        for sample_id, group in zip(sample_ids, groups):
            writer.writerow([sample_id, group, assignments[str(group)]])
    lines = ["| Representation | Mean R2 | 95% CI |", "|---|---:|---:|"]
    for method in METHODS:
        summary = aggregate[method]
        lines.append(f"| {method} | {summary['mean']:.4f} | [{summary['ci95'][0]:.4f}, {summary['ci95'][1]:.4f}] |")
    args.output_prefix.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
