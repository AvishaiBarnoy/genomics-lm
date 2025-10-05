"""Aggregate metrics across multiple runs."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List, Optional

from ._shared import RUNS_DIR, ensure_run_layout, read_meta

SUMMARY_COLUMNS = [
    "run_id",
    "val_ppl",
    "silhouette",
    "probe_aa",
    "probe_class_pol",
    "probe_class_hydro",
    "probe_is_stop",
    "probe_is_start",
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
    for line in path.read_text().splitlines():
        if line.lower().startswith("silhouette score:"):
            value = line.split(":", 1)[1].strip()
            if value.upper() == "NA":
                return None
            try:
                return float(value)
            except ValueError:
                return None
    return None


def _read_probe_results(run_dir: Path) -> dict:
    path = run_dir / "tables" / "probe_results.csv"
    if not path.exists():
        return {}
    results = {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            task = row.get("task")
            if not task:
                continue
            try:
                score = float(row.get("mean_accuracy", ""))
            except ValueError:
                continue
            results[task] = score
    return results


def main(argv: Optional[Iterable[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_ids", nargs="+")
    args = ap.parse_args(argv)

    summary_dir = RUNS_DIR / "_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    for run_id in args.run_ids:
        try:
            paths = ensure_run_layout(run_id)
        except Exception:
            continue
        run_dir = paths["run"]
        row = {col: "NA" for col in SUMMARY_COLUMNS}
        row["run_id"] = run_id
        try:
            meta = read_meta(run_dir)
            if meta.get("val_ppl") is not None:
                row["val_ppl"] = f"{float(meta['val_ppl']):.4f}"
        except Exception:
            pass
        silhouette = _read_silhouette(run_dir)
        if silhouette is not None:
            row["silhouette"] = f"{silhouette:.4f}"
        probe_scores = _read_probe_results(run_dir)
        for task, column in TASK_TO_COLUMN.items():
            if task in probe_scores:
                row[column] = f"{probe_scores[task]:.4f}"
        rows.append(row)

    def sort_key(row: dict) -> float:
        try:
            return float(row.get("val_ppl", float("inf")))
        except ValueError:
            return float("inf")

    rows.sort(key=sort_key)

    out_path = summary_dir / "summary.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"[compare] wrote summary to {out_path}")


if __name__ == "__main__":
    main()

