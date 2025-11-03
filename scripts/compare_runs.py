"""Aggregate metrics across multiple runs.

Modes:
- With explicit run_ids: preserves legacy behavior; writes runs/_summary/summary.csv with probe results.
- With no args: scans outputs/scores/*/metrics.json and builds a compare report under outputs/scores/compare/ with CSV + plots.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List, Optional, Dict

from ._shared import RUNS_DIR, ensure_run_layout, read_meta
import torch
import json
import numpy as np
import matplotlib.pyplot as plt

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
    ap.add_argument("run_ids", nargs="*")
    args = ap.parse_args(argv)

    # New scan mode when no run_ids provided
    if not args.run_ids:
        repo = Path(__file__).resolve().parents[1]
        scores_root = repo / "outputs/scores"
        compare_dir = scores_root / "compare"
        compare_dir.mkdir(parents=True, exist_ok=True)
        rows: List[Dict] = []
        for run_dir in scores_root.iterdir():
            if not run_dir.is_dir() or run_dir.name == "compare":
                continue
            metrics_path = run_dir / "metrics.json"
            if not metrics_path.exists():
                continue
            metrics = json.loads(metrics_path.read_text())
            run_id = run_dir.name
            # Try to load dims from checkpoint cfg
            ckpt_cfg = {}
            ckpt = repo / "outputs/checkpoints" / run_id / "best.pt"
            if ckpt.exists():
                try:
                    state = torch.load(ckpt, map_location="cpu")
                    ckpt_cfg = state.get("cfg", {}) if isinstance(state, dict) else {}
                except Exception:
                    ckpt_cfg = {}
            row = {
                "run_id": run_id,
                "n_layer": ckpt_cfg.get("n_layer"),
                "n_head": ckpt_cfg.get("n_head"),
                "n_embd": ckpt_cfg.get("n_embd"),
                "epochs": ckpt_cfg.get("epochs"),
                "val_ppl": metrics.get("val_ppl"),
                "test_ppl": metrics.get("test_ppl"),
                "codon_corr": metrics.get("codon_corr"),
                "frameshift_delta": metrics.get("frameshift_delta"),
                "startstop_delta.start": metrics.get("startstop_delta.start"),
                "startstop_delta.stop": metrics.get("startstop_delta.stop"),
                "syn_gap": metrics.get("syn_gap"),
                "timestamp": metrics.get("timestamp"),
            }
            # n_params (optional): estimate if ckpt present
            try:
                if ckpt.exists() and ckpt_cfg.get("vocab_size"):
                    from src.codonlm.model_tiny_gpt import TinyGPT
                    m = TinyGPT(
                        ckpt_cfg["vocab_size"], ckpt_cfg["block_size"],
                        ckpt_cfg["n_layer"], ckpt_cfg["n_head"], ckpt_cfg["n_embd"],
                        dropout=ckpt_cfg.get("dropout", 0.0), use_checkpoint=False,
                        label_smoothing=ckpt_cfg.get("label_smoothing", 0.0),
                    )
                    row["n_params"] = sum(p.numel() for p in m.parameters())
                else:
                    row["n_params"] = None
            except Exception:
                row["n_params"] = None
            rows.append(row)

        # write CSV
        cols = [
            "run_id", "n_params", "n_layer", "n_head", "n_embd", "epochs",
            "val_ppl", "test_ppl", "codon_corr", "frameshift_delta",
            "startstop_delta.start", "startstop_delta.stop", "syn_gap", "timestamp",
        ]
        out_csv = compare_dir / "summary.csv"
        with out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"[compare] wrote {out_csv}")

        # plots
        try:
            import matplotlib.pyplot as plt
            # ppl_vs_params
            xs = [r["n_params"] for r in rows if r.get("n_params") and r.get("test_ppl")]
            ys = [r["test_ppl"] for r in rows if r.get("n_params") and r.get("test_ppl")]
            if xs and ys:
                plt.figure()
                plt.scatter(xs, ys)
                plt.xlabel("n_params"); plt.ylabel("test_ppl"); plt.title("Test PPL vs Params")
                plt.tight_layout(); plt.savefig(compare_dir / "ppl_vs_params.png"); plt.close()
            # val_vs_test
            xs = [r["val_ppl"] for r in rows if r.get("val_ppl") and r.get("test_ppl")]
            ys = [r["test_ppl"] for r in rows if r.get("val_ppl") and r.get("test_ppl")]
            if xs and ys:
                plt.figure()
                plt.scatter(xs, ys)
                plt.xlabel("val_ppl"); plt.ylabel("test_ppl"); plt.title("Val vs Test PPL")
                plt.tight_layout(); plt.savefig(compare_dir / "val_vs_test_ppl.png"); plt.close()
        except Exception as exc:
            print(f"[compare] plotting failed: {exc}")
        return

    # Legacy explicit-run mode below
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
