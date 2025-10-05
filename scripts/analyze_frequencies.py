"""Frequency analysis for a collected run."""
from __future__ import annotations

import argparse
import csv
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ._shared import ArtifactError, ensure_run_layout, load_artifacts, load_token_list


def _top_k_indices(counts: np.ndarray, k: int = 20) -> np.ndarray:
    k = min(k, counts.shape[0])
    return np.argsort(counts)[::-1][:k]


def main(argv: Optional[Iterable[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    args = ap.parse_args(argv)

    paths = ensure_run_layout(args.run_id)
    run_dir, charts_dir, tables_dir = paths["run"], paths["charts"], paths["tables"]

    try:
        artifacts = load_artifacts(args.run_id)
    except ArtifactError as exc:
        print(f"[freq] {exc}")
        return

    tokens = load_token_list(run_dir)
    counts = artifacts.get("token_counts")
    first_counts = artifacts.get("first_token_counts")
    if counts is None or counts.size == 0:
        print("[freq] token_counts missing; nothing to plot")
        return

    total = counts.sum()
    freq_rows = []
    for idx, count in enumerate(counts):
        token = tokens[idx] if idx < len(tokens) else f"tok_{idx}"
        freq = (float(count) / float(total)) if total > 0 else 0.0
        freq_rows.append((token, int(count), freq))

    freq_path = tables_dir / "frequencies.csv"
    with freq_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["token", "count", "frequency"])
        writer.writerows(freq_rows)

    top_idx = _top_k_indices(counts, 20)
    top_tokens = [tokens[i] if i < len(tokens) else f"tok_{i}" for i in top_idx]
    top_counts = counts[top_idx]

    plt.figure(figsize=(10, 5))
    plt.bar(top_tokens, top_counts)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Count")
    plt.title("Top-20 token frequency")
    plt.tight_layout()
    plt.savefig(charts_dir / "top20_freq.png", dpi=200)
    plt.close()

    if first_counts is not None and first_counts.size == counts.size:
        plt.figure(figsize=(10, 5))
        plt.bar(np.arange(first_counts.size), first_counts)
        plt.xticks(range(len(tokens)), tokens, rotation=45, ha="right")
        plt.ylabel("Count")
        plt.title("First-position token counts")
        plt.tight_layout()
        plt.savefig(charts_dir / "first_position_counts.png", dpi=200)
        plt.close()
    else:
        print("[freq] first_token_counts missing; skipping plot")

    print(f"[freq] wrote tables to {freq_path}")


if __name__ == "__main__":
    main()

