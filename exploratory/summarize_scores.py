#!/usr/bin/env python3
import argparse, os, pandas as pd, numpy as np, matplotlib.pyplot as plt

CODONS = [a+b+c for a in "ACGT" for b in "ACGT" for c in "ACGT"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True, help="score_mutations TSV")
    ap.add_argument("--outdir", default="analysis/out")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.tsv, sep="\t")
    codon_cols = [c for c in df.columns if c in CODONS]
    # summaries
    df["best_delta"] = df[codon_cols].max(axis=1)
    df["n_better"]   = (df[codon_cols] > 0).sum(axis=1)

    # top-k table per position
    tops = []
    for _,row in df.iterrows():
        pos, wt = int(row["pos"]), row["wt"]
        deltas = row[codon_cols].astype(float)
        top = deltas.sort_values(ascending=False).head(args.topk)
        for cod, d in top.items():
            tops.append({"pos":pos, "wt":wt, "mut":cod, "delta":float(d)})
    topk_df = pd.DataFrame(tops)

    # write CSVs
    df[["pos","wt","best_delta","n_better"]].to_csv(os.path.join(args.outdir,"summary.csv"), index=False)
    topk_df.to_csv(os.path.join(args.outdir,"topk.csv"), index=False)

    # quick plots
    plt.figure()
    df["best_delta"].plot(title="Max ΔlogP per position")
    plt.xlabel("position"); plt.ylabel("ΔlogP"); plt.tight_layout()
    plt.savefig(os.path.join(args.outdir,"best_delta.png")); plt.close()

    plt.figure()
    df["n_better"].plot(title="# mutants with ΔlogP>0 per position")
    plt.xlabel("position"); plt.ylabel("count"); plt.tight_layout()
    plt.savefig(os.path.join(args.outdir,"n_better.png")); plt.close()

    # heatmap (may be large)
    plt.figure(figsize=(12,4))
    plt.imshow(df[codon_cols].to_numpy(), aspect="auto")
    plt.title("ΔlogP heatmap (pos × codon)"); plt.xlabel("codon"); plt.ylabel("position")
    plt.tight_layout(); plt.savefig(os.path.join(args.outdir,"heatmap.png")); plt.close()

    print(f"[save] {args.outdir}/summary.csv, topk.csv, best_delta.png, n_better.png, heatmap.png")

if __name__ == "__main__":
    main()

