import pandas as pd
import numpy as np
from pathlib import Path


def audit_dataset():
    meta_path = Path("data/processed/stage2_meta.tsv")
    if not meta_path.exists():
        print("[!] Meta file not found.")
        return

    df = pd.read_csv(meta_path, sep="\t")
    print("=== Stage 2 Dataset Audit ===")
    print(f"Total Sequences: {len(df)}")

    # Check Genome distribution
    counts = df["genome"].value_counts()
    print("\nGenome Distribution (Top 10):")
    print(counts.head(10))

    # Check if we have the new groups
    # We'll use the counts to show diversity
    print(f"\nUnique Genomes Count: {len(counts)}")

    # Identify which ones are from which folder by comparing against data/raw
    gram_pos_files = [
        Path(f).stem.split("_")[:2] for f in Path("data/raw/gram_pos").glob("*.gbff")
    ]
    gram_pos_ids = ["_".join(p) for p in gram_pos_files]

    high_gc_files = [
        Path(f).stem.split("_")[:2] for f in Path("data/raw/high-gc").glob("*.gbff")
    ]
    high_gc_ids = ["_".join(p) for p in high_gc_files]

    gram_pos_count = df[df["genome"].isin(gram_pos_ids)].shape[0]
    high_gc_count = df[df["genome"].isin(high_gc_ids)].shape[0]
    entero_count = len(df) - gram_pos_count - high_gc_count

    print("\nDiversity Metrics:")
    print(f"- Gram-positive sequences: {gram_pos_count}")
    print(f"- High-GC sequences: {high_gc_count}")
    print(f"- Enterobacteriaceae sequences: {entero_count}")

    # Check the packed NPZ files
    train_path = Path("data/processed/stage2_diverse/train_bs256.npz")
    if train_path.exists():
        with np.load(train_path) as data:
            X = data["X"]
            print(f"\nPacked Training Data Shape: {X.shape}")
            print(f"Total Training Windows: {X.shape[0]}")


if __name__ == "__main__":
    audit_dataset()
