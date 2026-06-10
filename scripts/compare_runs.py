"""Aggregate metrics across multiple runs.

Usage:
  python -m scripts.compare_runs [run_ids ...]
  python -m scripts.compare_runs --prefix 2025-10
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional


# Absolute imports to avoid parent package issues
from scripts._shared import RUNS_DIR, read_meta

SUMMARY_COLUMNS = [
    "run_id",
    "val_ppl",
    "silhouette",
    "probe_aa",
    "probe_class_pol",
    "probe_class_hydro",
    "probe_is_stop",
    "probe_is_start",
    "bio_recall",
]

TASK_TO_COLUMN = {
    "AA identity": "probe_aa",
    "polarity class": "probe_class_pol",
    "hydropathy class": "probe_class_hydro",
    "is_stop": "probe_is_stop",
    "is_start": "probe_is_start",
}


def _read_silhouette(run_dir: Path) -> Optional[float]:
    path = run_dir / "tables" / "embed_quality.txt"
    if not path.exists():
        return None
    try:
        content = path.read_text()
        for line in content.splitlines():
            if line.lower().startswith("silhouette score:"):
                value = line.split(":", 1)[1].strip()
                if value.upper() == "NA":
                    return None
                return float(value)
    except Exception:
        pass
    return None


def _read_probe_results(run_dir: Path) -> dict:
    path = run_dir / "tables" / "probe_results.csv"
    if not path.exists():
        return {}
    results = {}
    try:
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                task = row.get("task")
                if not task:
                    continue
                try:
                    score = float(row.get("mean_accuracy", ""))
                    results[task] = score
                except ValueError:
                    continue
    except Exception:
        pass
    return results


def _read_bio_recall(run_dir: Path) -> Optional[float]:
    path = run_dir / "motif_mining" / "biological_benchmark.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return data.get("biological_recall_score")
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_ids", nargs="*", help="Specific run IDs to compare")
    ap.add_argument("--prefix", help="Filter runs by prefix (e.g. '2025-10')")
    ap.add_argument(
        "--out", default="runs/_summary/summary.csv", help="Output CSV path"
    )
    args = ap.parse_args()

    run_ids = args.run_ids
    if not run_ids:
        # Scan RUNS_DIR
        if not RUNS_DIR.exists():
            print(f"[!] {RUNS_DIR} does not exist.")
            return

        run_ids = [
            d.name
            for d in RUNS_DIR.iterdir()
            if d.is_dir() and not d.name.startswith("_")
        ]

        if args.prefix:
            run_ids = [r for r in run_ids if r.startswith(args.prefix)]

    if not run_ids:
        print("[!] No runs found to compare.")
        return

    rows = []
    for run_id in run_ids:
        run_dir = RUNS_DIR / run_id
        if not run_dir.exists():
            continue

        row = {col: "NA" for col in SUMMARY_COLUMNS}
        row["run_id"] = run_id

        # 1. Val PPL from meta
        try:
            meta = read_meta(run_dir)
            val_ppl = meta.get("val_ppl", meta.get("last_perplexity"))
            if val_ppl is not None:
                row["val_ppl"] = f"{float(val_ppl):.4f}"
        except Exception:
            pass

        # 2. Silhouette
        sil = _read_silhouette(run_dir)
        if sil is not None:
            row["silhouette"] = f"{sil:.4f}"

        # 3. Probes
        probes = _read_probe_results(run_dir)
        for task, col in TASK_TO_COLUMN.items():
            if task in probes:
                row[col] = f"{probes[task]:.4f}"

        # 4. Biological Recall (New!)
        recall = _read_bio_recall(run_dir)
        if recall is not None:
            row["bio_recall"] = f"{recall:.4f}"

        rows.append(row)

    # Sort by val_ppl (lowest first)
    def sort_key(r):
        try:
            return float(r["val_ppl"])
        except ValueError:
            return float("inf")

    rows.sort(key=sort_key)

    # Write output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[success] Wrote summary of {len(rows)} runs to {out_path}")


if __name__ == "__main__":
    main()
