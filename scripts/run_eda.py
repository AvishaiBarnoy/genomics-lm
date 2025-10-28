#!/usr/bin/env python3
"""
Run exploratory EDA utilities for a completed run.

This wrapper:
  - creates runs/<RUN_ID>/analysis/eda
  - if motif clusters exist, runs exploratory/inspect_motifs.py and stores tables there
  - if one_cds__best.tsv exists, runs exploratory/summarize_scores.py and stores summaries and plots
  - optionally writes a quick text summary via exploratory/eda_stub.py

Usage:
  python -m scripts.run_eda <RUN_ID> [--config CONFIG] [--k 9] [--samples 20000]
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def _call(cmd: list[str]) -> None:
    print("[eda] run:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--config", default="configs/tiny_mps.yaml")
    ap.add_argument("--k", type=int, default=9)
    ap.add_argument("--samples", type=int, default=20000)
    ap.add_argument("--stub", action="store_true", help="also emit a small text EDA summary")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    run_dir = repo / "runs" / args.run_id
    eda_dir = run_dir / "exploratory" / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)

    # Resolve combined manifest/train npz
    manifest = None
    manifest_path = run_dir / "combined_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        alt = repo / "data" / "processed" / "combined" / args.run_id / "manifest.json"
        if alt.exists():
            manifest = json.loads(alt.read_text())
    train_npz = manifest.get("train") if manifest else None

    # Motif clusters
    clusters_file = None
    # Prefer per-run copy
    cand = run_dir / "motif_clusters.npz"
    if cand.exists():
        clusters_file = str(cand)
    else:
        cand2 = repo / "outputs" / "motif_clusters.npz"
        if cand2.exists():
            clusters_file = str(cand2)

    if clusters_file and train_npz:
        motifs_out = eda_dir / "motifs"
        motifs_out.mkdir(parents=True, exist_ok=True)
        _call([
            "python", str(repo / "exploratory" / "inspect_motifs.py"),
            "--clusters_npz", clusters_file,
            "--train_npz", train_npz,
            "--k", str(args.k),
            "--samples", str(args.samples),
            "--outdir", str(motifs_out),
        ])
    else:
        if not clusters_file:
            print("[eda] skip motifs: motif_clusters.npz not found")
        if not train_npz:
            print("[eda] skip motifs: combined train NPZ not found")

    # One-CDS mutation scores
    one_out = repo / "outputs" / "scores" / args.run_id / "one_cds__best.tsv"
    if one_out.exists():
        scores_out = eda_dir / "one_cds"
        scores_out.mkdir(parents=True, exist_ok=True)
        _call([
            "python", str(repo / "exploratory" / "summarize_scores.py"),
            "--tsv", str(one_out),
            "--outdir", str(scores_out),
        ])
        if args.stub:
            # Also write quick text summary using the stub
            txt = scores_out / "eda_quick.txt"
            with txt.open("w") as fh:
                subprocess.run([
                    "python", str(repo / "exploratory" / "eda_stub.py"),
                    "--tsv", str(one_out),
                ], check=True, stdout=fh)
    else:
        print("[eda] skip summarize_scores: one_cds__best.tsv not found under outputs/scores/", args.run_id)

    print(f"[eda] results in {eda_dir}")


if __name__ == "__main__":
    main()

