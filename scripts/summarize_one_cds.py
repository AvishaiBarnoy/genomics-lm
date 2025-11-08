"""Summarize one_cds__best.tsv if present."""
from __future__ import annotations

import argparse
import csv
from typing import Iterable, Optional

import numpy as np

from ._shared import ensure_run_layout, resolve_run


def main(argv: Optional[Iterable[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id", nargs="?")
    ap.add_argument("--run_dir", help="Alternative to run_id; path to runs/<RUN_ID>")
    args = ap.parse_args(argv)

    run_id, run_dir = resolve_run(args.run_id, args.run_dir)
    paths = ensure_run_layout(run_id)
    run_dir, tables_dir = paths["run"], paths["tables"]

    source = run_dir / "one_cds__best.tsv"
    if not source.exists():
        print(f"[one-cds] {source} missing; skipping summary")
        return

    with source.open("r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
        headers = reader.fieldnames or []

    if not rows:
        print("[one-cds] no rows to summarize")
        return

    numeric_cols = []
    for h in headers:
        try:
            float(rows[0][h])
            numeric_cols.append(h)
        except (TypeError, ValueError):
            continue

    summaries = []
    for col in numeric_cols:
        values = []
        for row in rows:
            try:
                values.append(float(row[col]))
            except (TypeError, ValueError):
                continue
        if not values:
            continue
        arr = np.array(values, dtype=np.float64)
        summaries.append(
            {
                "metric": col,
                "count": arr.size,
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                "min": float(arr.min()),
                "max": float(arr.max()),
            }
        )

    if not summaries:
        print("[one-cds] no numeric columns found")
        return

    out_path = tables_dir / "one_cds__summary.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "count", "mean", "std", "min", "max"])
        for row in summaries:
            writer.writerow(
                [
                    row["metric"],
                    row["count"],
                    f"{row['mean']:.4f}",
                    f"{row['std']:.4f}",
                    f"{row['min']:.4f}",
                    f"{row['max']:.4f}",
                ]
            )

    print(f"[one-cds] wrote summary to {out_path}")


if __name__ == "__main__":
    main()
