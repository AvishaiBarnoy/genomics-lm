#!/usr/bin/env python3
"""
Wrapper around src.codonlm.score_mutations to produce a per-position ΔlogP CSV and a heatmap PNG.

Usage:
  python -m scripts.infer_score_mutations --run_dir outputs/checkpoints/<RUN_ID> --seq "ATG ... TGA" --out_dir outputs/analysis/<RUN_ID>
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--seq", required=True, help="space-separated codons from ATG ... STOP")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Write temporary DNA file (single CDS without spaces)
    dna = args.seq.replace(" ", "").upper()
    tmp = out_dir / "_one_cds.txt"
    tmp.write_text(dna + "\n")

    ckpt = run_dir / "best.pt"
    tsv = out_dir / "mut_scores.tsv"
    subprocess.run([
        "python", "-m", "src.codonlm.score_mutations",
        "--ckpt", str(ckpt),
        "--dna", str(tmp),
        "--out", str(tsv),
    ], check=True)

    # Plot simple heatmap from TSV (pos × codon)
    df = pd.read_csv(tsv, sep="\t")
    codons = [c for c in df.columns if len(c) == 3]
    plt.figure(figsize=(12, 4))
    plt.imshow(df[codons].to_numpy(), aspect="auto")
    plt.xlabel("codon"); plt.ylabel("position"); plt.title("ΔlogP heatmap")
    plt.tight_layout(); plt.savefig(out_dir / "mut_scores_heatmap.png"); plt.close()
    print(f"[mut] wrote {tsv} and heatmap.png")


if __name__ == "__main__":
    main()

