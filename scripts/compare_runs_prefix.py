#!/usr/bin/env python3
"""Compare prefix-generation summaries across runs.

CLI:
  --run_ids A,B,C
  --metrics median_gqs,median_gqs_norm,mean_aa_identity,best_aa_identity,terminal_stop_rate
  --out_dir outputs/figs
Writes a combined CSV and a simple facet-like plot (one line per run) for selected metrics vs k.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt


def read_summary(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with path.open("r", newline="") as f:
        for row in csv.DictReader(f):
            try:
                row["k"] = int(row["k"])
            except Exception:
                continue
            rows.append({k: (float(v) if k != "k" else row["k"]) for k, v in row.items() if v != ""})
    rows.sort(key=lambda r: r["k"])
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_ids", required=True, help="comma-separated run IDs")
    ap.add_argument("--metrics", default="median_gqs,mean_aa_identity",
                    help="comma-separated columns in summary.csv to plot")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    combine_csv = out_dir / "compare_runs_prefix.csv"
    runs = [r.strip() for r in args.run_ids.split(",") if r.strip()]
    # Load summaries
    summaries: Dict[str, List[Dict[str, float]]] = {}
    for run_id in runs:
        path = repo / "outputs" / "scores" / run_id / "gen_prefix" / "summary.csv"
        if path.exists():
            summaries[run_id] = read_summary(path)

    # Write combined CSV (inner-join on k)
    ks = sorted(set.intersection(*[set(r["k"] for r in rows) for rows in summaries.values()])) if summaries else []
    with combine_csv.open("w", newline="") as f:
        fieldnames = ["k"] + [f"{run_id}:{m}" for run_id in summaries for m in metrics]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for k in ks:
            row = {"k": k}
            for run_id, rows in summaries.items():
                rowk = next((r for r in rows if r["k"] == k), None)
                if rowk:
                    for m in metrics:
                        if m in rowk:
                            row[f"{run_id}:{m}"] = rowk[m]
            writer.writerow(row)

    # Plot: one figure with lines per run for first metric
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(111)
    for run_id, rows in summaries.items():
        xs = [r["k"] for r in rows]
        ys = [r.get(metrics[0]) for r in rows]
        ax.plot(xs, ys, marker="o", label=run_id)
    ax.set_xlabel("k"); ax.set_ylabel(metrics[0]); ax.legend()
    fig.tight_layout(); fig.savefig(out_dir / "compare_runs_prefix.png"); plt.close(fig)
    print(f"[compare-prefix] wrote {combine_csv} and {out_dir / 'compare_runs_prefix.png'}")


if __name__ == "__main__":
    main()

