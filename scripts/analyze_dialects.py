import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Standard Codon -> AA mapping
CODON_TO_AA = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M', 'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K', 'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L', 'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q', 'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V', 'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E', 'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S', 'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_', 'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W',
}

def analyze_dialects():
    dna_path = Path("data/processed/stage2_dna.txt")
    meta_path = Path("data/processed/stage2_meta.tsv")
    
    if not dna_path.exists() or not meta_path.exists():
        print("[!] Stage 2 data files not found.")
        return

    print("[*] Loading Stage 2 Data...")
    df_meta = pd.read_csv(meta_path, sep="\t")
    with open(dna_path) as f:
        sequences = [line.strip() for line in f]
    
    # Identify groups
    gram_pos_ids = ["_".join(Path(f).stem.split("_")[:2]) for f in Path("data/raw/gram_pos").glob("*.gbff")]
    high_gc_ids = ["_".join(Path(f).stem.split("_")[:2]) for f in Path("data/raw/high-gc").glob("*.gbff")]
    
    def get_group(genome):
        if genome in gram_pos_ids: return "Gram-positive"
        if genome in high_gc_ids: return "High-GC"
        return "Enterobacteriaceae"

    df_meta['group'] = df_meta['genome'].apply(get_group)
    
    # Calculate Codon Usage per Group
    group_counts = {g: Counter() for g in df_meta['group'].unique()}
    
    print("[*] Counting codons by group...")
    for idx, row in df_meta.iterrows():
        seq = sequences[idx]
        group = row['group']
        for i in range(0, len(seq)-2, 3):
            codon = seq[i:i+3]
            if len(codon) == 3:
                group_counts[group][codon] += 1

    # Convert to Frequencies
    data = []
    for group, counts in group_counts.items():
        total = sum(counts.values())
        for codon, count in counts.items():
            data.append({
                'group': group,
                'codon': codon,
                'aa': CODON_TO_AA.get(codon, '?'),
                'freq': count / total
            })
    
    df_usage = pd.DataFrame(data)
    
    # Visualization: Top Differences
    # Pivot to see freq per group
    pivot = df_usage.pivot(index='codon', columns='group', values='freq').fillna(0)
    
    # Calculate Max Difference (Range) across groups for each codon
    pivot['variance'] = pivot.max(axis=1) - pivot.min(axis=1)
    top_diff = pivot.sort_values('variance', ascending=False).head(15)
    
    print("\n=== Top Codon Usage Differences (Dialects) ===")
    print(top_diff)
    
    # Plotting
    out_dir = Path("outputs/analysis/stage2_diversity")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    top_diff.drop('variance', axis=1).plot(kind='bar', figsize=(14, 7))
    plt.title("Genomic Dialects: Major Codon Usage Differences in Stage 2 Dataset")
    plt.ylabel("Frequency")
    plt.xlabel("Codon")
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(out_dir / "dialect_comparison.png")
    
    print(f"\n[success] Dialect analysis saved to {out_dir}/dialect_comparison.png")

if __name__ == "__main__":
    analyze_dialects()
