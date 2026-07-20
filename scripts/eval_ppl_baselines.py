#!/usr/bin/env python3
"""Evaluate uniform and count-based next-token baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.codonlm.data_loading import MmapPackedDataset
from src.codonlm.dataset_manifest import load_dataset_manifest, manifest_artifact_path
from src.codonlm.training.vocabulary import resolve_vocabulary_contract

PAD_ID = 0
MODEL_NAMES = ("Uniform", "Unigram", "Bigram", "Trigram")


def _examples(path: Path):
    """Yield (inputs, targets) using the same target construction as training."""
    dataset = MmapPackedDataset(path)
    for index in range(len(dataset)):
        item = dataset[index]
        if dataset.is_dynamic:
            sequence = np.asarray(item)
            yield sequence[:-1], sequence[1:]
        else:
            x, y = item
            yield np.asarray(x), np.asarray(y)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_hashes(path: Path) -> dict[str, str]:
    files = [path]
    files.extend(
        candidate
        for suffix in ("_X.npy", "_Y.npy", "_lengths.npy")
        if (candidate := path.with_name(path.stem + suffix)).exists()
    )
    return {str(file.resolve()): _hash(file) for file in files if file.exists()}


def fit_baselines(train_path: Path, vocab_size: int, alpha: float = 0.01):
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    unigram = np.zeros(vocab_size, dtype=np.int64)
    bigram = defaultdict(lambda: np.zeros(vocab_size, dtype=np.int64))
    trigram = defaultdict(lambda: np.zeros(vocab_size, dtype=np.int64))
    for x, y in _examples(train_path):
        for position, (previous, target) in enumerate(zip(x, y)):
            previous, target = int(previous), int(target)
            if target == PAD_ID:
                continue
            unigram[target] += 1
            bigram[previous][target] += 1
            previous2 = int(x[position - 1]) if position else PAD_ID
            trigram[(previous2, previous)][target] += 1
    if int(unigram.sum()) == 0:
        raise ValueError(f"training dataset has no evaluable non-PAD targets: {train_path}")
    return unigram, dict(bigram), dict(trigram)


def _probability(counts, target: int, alpha: float, active_size: int) -> float:
    total = float(np.asarray(counts)[1:].sum()) if counts is not None else 0.0
    count = float(counts[target]) if counts is not None else 0.0
    return (count + alpha) / (total + alpha * active_size)


def evaluate_baselines(test_path, counts, vocab_size: int, alpha: float = 0.01):
    unigram, bigram, trigram = counts
    active_size = vocab_size - 1
    nll = {name: 0.0 for name in MODEL_NAMES}
    tokens = 0
    for x, y in _examples(test_path):
        for position, (previous, target) in enumerate(zip(x, y)):
            previous, target = int(previous), int(target)
            if target == PAD_ID:
                continue
            tokens += 1
            previous2 = int(x[position - 1]) if position else PAD_ID
            nll["Uniform"] += math.log(active_size)
            nll["Unigram"] -= math.log(_probability(unigram, target, alpha, active_size))
            nll["Bigram"] -= math.log(
                _probability(bigram.get(previous), target, alpha, active_size)
            )
            tri_counts = trigram.get((previous2, previous))
            if tri_counts is None:
                tri_counts = bigram.get(previous)
            nll["Trigram"] -= math.log(
                _probability(tri_counts, target, alpha, active_size)
            )
    if tokens == 0:
        raise ValueError(f"test dataset has no evaluable non-PAD targets: {test_path}")
    results = {}
    for name in MODEL_NAMES:
        loss = nll[name] / tokens
        results[name] = {
            "cross_entropy_nats": loss,
            "perplexity": math.exp(loss),
            "bits_per_codon": loss / math.log(2),
        }
    best_name = min((name for name in MODEL_NAMES if name != "Uniform"), key=lambda n: results[n]["cross_entropy_nats"])
    best = results[best_name]["cross_entropy_nats"]
    for metrics in results.values():
        metrics["cross_entropy_improvement_over_best_simple"] = best - metrics["cross_entropy_nats"]
    return results, tokens, best_name


def _markdown(report: dict) -> str:
    lines = [
        "| Model | Cross-entropy (nats) | Perplexity | Bits/codon | Improvement over best simple (nats) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, metrics in report["results"].items():
        lines.append(
            f"| {name} | {metrics['cross_entropy_nats']:.6f} | {metrics['perplexity']:.6f} | "
            f"{metrics['bits_per_codon']:.6f} | {metrics['cross_entropy_improvement_over_best_simple']:.6f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", "--train_npz", dest="train", required=True)
    parser.add_argument("--test", "--test_npz", dest="test", required=True)
    parser.add_argument("--itos", help="Vocabulary artifact; defaults to dataset-adjacent itos.txt")
    parser.add_argument("--manifest", type=Path, help="Dataset manifest to hash into provenance")
    parser.add_argument("--config", type=Path, help="Evaluation/run config to hash into provenance")
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--output-prefix", type=Path)
    args = parser.parse_args()
    train, test = Path(args.train), Path(args.test)
    manifest_provenance = {"status": "legacy_unverified"}
    if args.manifest is not None:
        manifest = load_dataset_manifest(args.manifest)
        for split, selected in (("train", train), ("test", test)):
            declared = manifest_artifact_path(
                manifest, args.manifest.resolve(), f"{split}_tokens"
            ).resolve()
            if selected.resolve() != declared:
                raise ValueError(
                    f"{split} dataset {selected.resolve()} does not match manifest artifact {declared}"
                )
        manifest_provenance = {
            "path": str(args.manifest.resolve()),
            "dataset_id": manifest["dataset"]["id"],
            "scientific_valid": manifest["dataset"]["scientific_valid"],
            "schema": manifest["schema"],
        }
    contract = resolve_vocabulary_contract(
        [train, test], configured_path=args.itos, configured_size=None
    )
    counts = fit_baselines(train, contract.size, args.alpha)
    results, tokens, best = evaluate_baselines(test, counts, contract.size, args.alpha)
    report = {
        "schema_version": 1,
        "train": str(train.resolve()),
        "test": str(test.resolve()),
        "dataset_sha256": {**_artifact_hashes(train), **_artifact_hashes(test)},
        "input_provenance": {
            name: {"path": str(path.resolve()), "sha256": _hash(path)}
            for name, path in (("manifest", args.manifest), ("config", args.config))
            if path is not None
        },
        "dataset_manifest": manifest_provenance,
        "vocabulary": contract.provenance(),
        "smoothing": {"method": "additive", "alpha": args.alpha},
        "evaluated_tokens": tokens,
        "best_simple_baseline": best,
        "results": results,
    }
    markdown = _markdown(report)
    print(f"Baseline Perplexity Comparison ({tokens} tokens)\n{markdown}")
    prefix = args.output_prefix or test.with_name(test.stem + "_ppl_baselines")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    prefix.with_suffix(".md").write_text(markdown)


if __name__ == "__main__":
    main()
