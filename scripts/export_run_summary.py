#!/usr/bin/env python3
"""
Aggregate key run artifacts into a compact JSON summary that is easy for an LLM
to consume.

Usage:
    python -m scripts.export_run_summary RUN_ID [--top 8]
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from ._shared import ensure_run_layout, read_meta, resolve_run


@dataclass
class RunSummaryConfig:
    run_id: str
    top_n: int = 10
    output: Optional[Path] = None
    include_tables: bool = True


def _read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        return [row for row in reader]


def _load_top_frequencies(path: Path, top_n: int) -> List[Dict[str, object]]:
    rows = _read_csv_dicts(path)
    parsed = []
    for row in rows:
        try:
            count = int(row["count"])
            freq = float(row["frequency"])
        except (KeyError, ValueError):
            continue
        parsed.append({"token": row.get("token", ""), "count": count, "frequency": freq})
    parsed.sort(key=lambda x: x["count"], reverse=True)
    return parsed[:top_n]


def _load_nearest_neighbors(path: Path, top_n: int) -> List[Dict[str, object]]:
    rows = _read_csv_dicts(path)
    summary: List[Dict[str, object]] = []
    for row in rows[:top_n]:
        token = row.get("token", "")
        neighbors = []
        for i in range(1, 6):
            tok = row.get(f"neighbor_{i}")
            prob = row.get(f"sim_{i}")
            if tok:
                neighbors.append({"token": tok, "similarity": float(prob) if prob else None})
        summary.append({"token": token, "neighbors": neighbors})
    return summary


def _load_next_token_tests(path: Path) -> List[Dict[str, object]]:
    rows = _read_csv_dicts(path)
    summary: List[Dict[str, object]] = []
    for row in rows:
        prefix = row.get("prefix", "")
        preds = []
        for i in range(1, 6):
            token = row.get(f"pred_{i}")
            prob = row.get(f"prob_{i}")
            if token:
                preds.append({"token": token, "prob": float(prob) if prob else None})
        summary.append({"prefix": prefix, "predictions": preds})
    return summary


def _load_saliency(path: Path, top_n: int) -> List[Dict[str, object]]:
    rows = _read_csv_dicts(path)
    parsed = []
    for row in rows:
        try:
            pos = int(row["position"])
            val = float(row["saliency"])
        except (KeyError, ValueError):
            continue
        token = row.get("token", "")
        parsed.append({"position": pos, "token": token, "saliency": val, "abs_saliency": abs(val)})
    parsed.sort(key=lambda x: x["abs_saliency"], reverse=True)
    return [
        {"position": r["position"], "token": r["token"], "saliency": r["saliency"]}
        for r in parsed[:top_n]
    ]


def _load_probe_results(path: Path) -> List[Dict[str, object]]:
    rows = _read_csv_dicts(path)
    summary = []
    for row in rows:
        task = row.get("task")
        if not task:
            continue
        try:
            mean = float(row.get("mean_accuracy", ""))
        except (TypeError, ValueError):
            continue
        try:
            std = float(row.get("std_accuracy", ""))
        except (TypeError, ValueError):
            std = None
        summary.append({"task": task, "mean_accuracy": mean, "std_accuracy": std})
    return summary


def _load_embed_quality(path: Path) -> Dict[str, object]:
    metrics: Dict[str, object] = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().lower()
        values = []
        for part in value.split(","):
            part = part.strip()
            if not part or part.upper() == "NA":
                continue
            try:
                values.append(float(part))
            except ValueError:
                continue
        if not values:
            continue
        metrics[key] = values if len(values) > 1 else values[0]
    return metrics


def _list_files(path: Path, suffixes: Iterable[str]) -> List[str]:
    files: List[str] = []
    for suffix in suffixes:
        files.extend(sorted(p.name for p in path.glob(f"*{suffix}")))
    return files


def build_summary(config: RunSummaryConfig) -> Dict[str, object]:
    run_paths = ensure_run_layout(config.run_id)
    run_dir = run_paths["run"]
    tables_dir = run_paths["tables"]
    charts_dir = run_paths["charts"]

    meta = read_meta(run_dir)

    relative = lambda p: str(p.relative_to(run_dir))  # noqa: E731

    summary: Dict[str, object] = {
        "run_id": config.run_id,
        "meta": {
            "val_ppl": meta.get("val_ppl"),
            "checkpoint_path": meta.get("checkpoint_path"),
            "config_path": meta.get("config_path"),
            "token_count": meta.get("token_count"),
            "model_spec": meta.get("model_spec"),
        },
        "tables_used": [],
        "charts_available": _list_files(charts_dir, [".png"]),
    }

    freq_path = tables_dir / "frequencies.csv"
    if freq_path.exists():
        summary["frequency_top_tokens"] = _load_top_frequencies(freq_path, config.top_n)
        summary["tables_used"].append(relative(freq_path))

    nn_path = tables_dir / "nearest_neighbors.csv"
    if nn_path.exists():
        summary["embedding_neighbors"] = _load_nearest_neighbors(nn_path, config.top_n)
        summary["tables_used"].append(relative(nn_path))

    embed_quality_path = tables_dir / "embed_quality.txt"
    if embed_quality_path.exists():
        summary["embedding_quality"] = _load_embed_quality(embed_quality_path)
        summary["tables_used"].append(relative(embed_quality_path))

    saliency_path = tables_dir / "saliency.csv"
    if saliency_path.exists():
        summary["saliency_top_positions"] = _load_saliency(saliency_path, config.top_n)
        summary["tables_used"].append(relative(saliency_path))

    next_token_path = tables_dir / "next_token_tests.csv"
    if next_token_path.exists():
        summary["next_token_tests"] = _load_next_token_tests(next_token_path)
        summary["tables_used"].append(relative(next_token_path))

    probe_path = tables_dir / "probe_results.csv"
    if probe_path.exists():
        summary["probe_results"] = _load_probe_results(probe_path)
        summary["tables_used"].append(relative(probe_path))

    return summary


def run_export(config: RunSummaryConfig) -> Path:
    summary = build_summary(config)
    run_dir = (ensure_run_layout(config.run_id))["run"]
    output_path = config.output or (run_dir / "llm_summary.json")
    with output_path.open("w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    print(f"[export-run-summary] wrote {output_path}")
    return output_path


def parse_args() -> RunSummaryConfig:
    parser = argparse.ArgumentParser(description="Export run artifacts into a compact JSON summary.")
    parser.add_argument("run_id", nargs="?", help="Run identifier under runs/")
    parser.add_argument("--run_dir", help="Alternative to run_id; path to runs/<RUN_ID>")
    parser.add_argument("--top", type=int, default=10, help="Top-N items to keep from each table (default: 10)")
    parser.add_argument("--output", type=str, default=None, help="Optional custom output path")
    args = parser.parse_args()

    # resolve run_id if run_dir provided
    rid, _ = resolve_run(args.run_id, args.run_dir)
    output = Path(args.output) if args.output else None
    return RunSummaryConfig(run_id=rid, top_n=max(1, args.top), output=output)


def main() -> None:
    config = parse_args()
    run_export(config)


if __name__ == "__main__":
    main()
