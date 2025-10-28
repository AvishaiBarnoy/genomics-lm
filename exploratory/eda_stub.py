#!/usr/bin/env python3
"""
EDA stub: read a score_mutations TSV and print a few summaries.
Usage:
  python analysis/eda_stub.py --tsv outputs/scores/<RUN_ID>/EXAMPLE.tsv
"""
import argparse, pandas as pd

CODONS = [a+b+c for a in "ACGT" for b in "ACGT" for c in "ACGT"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.tsv, sep="\t")
    codon_cols = [c for c in df.columns if c in CODONS]
    df["best_delta"] = df[codon_cols].max(axis=1)
    df["n_better"] = (df[codon_cols] > 0).sum(axis=1)

    print(df[["pos","wt","best_delta","n_better"]].head())
    print("\nTop-5 positions by best_delta:")
    print(df.nlargest(5, "best_delta")[["pos","wt","best_delta"]])

if __name__ == "__main__":
    main()
