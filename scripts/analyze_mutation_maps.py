#!/usr/bin/env python3
"""
Analyze mutation maps (ΔlogP TSVs) to identify sensitive sites and preference patterns.
Fulfills Step 2 of the Genomics-LM interpretability pipeline.
"""

import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scripts._shared import resolve_run, ensure_run_layout

# Standard DNA translation table
CODON_TABLE = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
    'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
    'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
    'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
    'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
    'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
    'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
    'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_',
    'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W',
}

def analyze_tsv(tsv_path: Path, run_id: str = None):
    print(f"[*] Analyzing {tsv_path}...")
    df = pd.read_csv(tsv_path, sep="\t")
    
    # Identify codon columns
    codons = [c for c in df.columns if len(c) == 3 and c.isupper()]
    if not codons:
        print(f"[!] No codon columns found in {tsv_path}. Skipping.")
        return

    # Calculate metrics
    df['best_delta'] = df[codons].max(axis=1)
    df['n_better'] = (df[codons] > 0).sum(axis=1)
    
    # Identify synonymous vs non-synonymous
    def get_mut_stats(row):
        wt_aa = CODON_TABLE.get(row['wt'], '?')
        syn_better = 0
        nonsyn_better = 0
        for c in codons:
            if c == row['wt']: continue
            if row[c] > 0:
                if CODON_TABLE.get(c, '?') == wt_aa:
                    syn_better += 1
                else:
                    nonsyn_better += 1
        return pd.Series([syn_better, nonsyn_better])

    df[['n_better_syn', 'n_better_nonsyn']] = df.apply(get_mut_stats, axis=1)

    # Setup output
    if run_id:
        _, run_dir = resolve_run(run_id=run_id)
        layout = ensure_run_layout(run_id)
        out_dir = layout['charts']
        table_dir = layout['tables']
    else:
        out_dir = Path("outputs/analysis/mutation_maps") / tsv_path.stem
        table_dir = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

    # Export summary CSV
    summary_path = table_dir / f"{tsv_path.stem}_summary.csv"
    cols = ['pos', 'wt', 'best_delta', 'n_better', 'n_better_syn', 'n_better_nonsyn']
    df[cols].to_csv(summary_path, index=False)
    
    # Export Top-K mutants per position
    topk_list = []
    for idx, row in df.iterrows():
        # Get top 3 mutations for this position
        muts = row[codons].sort_values(ascending=False).head(4) # head 4 because WT=0 might be there
        for cod, val in muts.items():
            if cod == row['wt']: continue
            topk_list.append({
                'pos': row['pos'],
                'wt': row['wt'],
                'mutant': cod,
                'delta': val,
                'is_synonymous': CODON_TABLE.get(cod) == CODON_TABLE.get(row['wt'])
            })
    topk_df = pd.DataFrame(topk_list)
    topk_path = table_dir / f"{tsv_path.stem}_top_mutants.csv"
    topk_df.to_csv(topk_path, index=False)

    # Visualization
    # 1. Heatmap
    plt.figure(figsize=(24, 12))
    sns.heatmap(df[codons].T, cmap="RdBu_r", center=0, cbar_kws={'label': 'ΔlogP'})
    plt.title(f"Mutation Map (ΔlogP relative to WT)\n{tsv_path.name}")
    plt.xlabel("Sequence Position")
    plt.ylabel("Codon Substitution")
    plt.tight_layout()
    plt.savefig(out_dir / f"{tsv_path.stem}_heatmap.png", dpi=150)
    plt.close()

    # 2. Per-position Best Delta
    plt.figure(figsize=(15, 6))
    plt.plot(df['pos'], df['best_delta'], color='steelblue', label='Max ΔlogP')
    plt.fill_between(df['pos'], 0, df['best_delta'], color='steelblue', alpha=0.2)
    plt.axhline(0, color='red', linestyle='--', alpha=0.5)
    plt.title(f"Highest Predicted Gain per Position\n{tsv_path.name}")
    plt.xlabel("Position")
    plt.ylabel("max ΔlogP")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"{tsv_path.stem}_profile.png")
    plt.close()

    print(f"[success] Results saved to {out_dir}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", help="Path to a single mutation TSV")
    ap.add_argument("--run_id", help="Analyze all mutation maps for a specific run_id")
    args = ap.parse_args()

    if args.tsv:
        analyze_tsv(Path(args.tsv), run_id=args.run_id)
    elif args.run_id:
        _, run_dir = resolve_run(run_id=args.run_id)
        # Search for .tsv files in the run directory
        tsvs = list(run_dir.glob("*.tsv"))
        if not tsvs:
            # Also check outputs/scores/RUN_ID
            score_dir = Path("outputs/scores") / args.run_id
            if score_dir.exists():
                tsvs = list(score_dir.glob("*.tsv"))
        
        if not tsvs:
            print(f"[!] No .tsv files found for run {args.run_id}")
            return
            
        for tsv in tsvs:
            analyze_tsv(tsv, run_id=args.run_id)
    else:
        print("[!] Please provide either --tsv or --run_id")

if __name__ == "__main__":
    main()
