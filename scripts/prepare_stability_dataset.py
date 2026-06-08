#!/usr/bin/env python3
"""
Stage 3: Stability Dataset Preparation
Converts MegaScale experimental data into JSONL for ProteinLM training.
"""

import pandas as pd
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

def prepare_stability_data(csv_path, out_dir):
    print(f"[*] Loading stability data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Required columns: sequence, label
    # MegaScale Fig 5 has 'aa_seq' and 'deltaG'
    if 'aa_seq' not in df.columns or 'deltaG' not in df.columns:
        print("[!] Missing required columns. Checking other Fig 1 format...")
        # Fallback for Fig 1 which uses different names sometimes?
        # No, Fig 1 also used aa_seq and deltaG.
        return

    # Filter out NaNs
    df = df.dropna(subset=['aa_seq', 'deltaG'])
    
    # Create stability labels (Binary classification for the Expert)
    # Threshold 3.0 based on median/mean distribution
    df['stability_label'] = df['deltaG'].apply(lambda x: 'stable' if x >= 3.0 else 'unstable')
    
    # Convert to JSONL format
    samples = []
    for _, row in df.iterrows():
        samples.append({
            "sequence": row['aa_seq'],
            "stability_score": float(row['deltaG']),
            "stability_label": row['stability_label'],
            "source": "MegaScale_Tsuboyama_2023"
        })
    
    # Split into train/val
    train, val = train_test_split(samples, test_size=0.1, random_state=42)
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "stability_train.jsonl", "w") as f:
        for s in train:
            f.write(json.dumps(s) + "\n")
            
    with open(out_dir / "stability_val.jsonl", "w") as f:
        for s in val:
            f.write(json.dumps(s) + "\n")
            
    print(f"[success] Prepared {len(train)} training and {len(val)} validation samples.")
    print(f"[save] Files saved to {out_dir}")

if __name__ == "__main__":
    prepare_stability_data(
        "data/raw/stability/dG_extdG_data_Fig1.csv",
        "data/processed/protein_lm/stability"
    )
