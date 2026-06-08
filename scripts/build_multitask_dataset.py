#!/usr/bin/env python3
"""
Stage 3: Build Multi-Task Protein Dataset
Merges UniProt metadata (Pfam, EC) and MegaScale stability data into a unified JSONL format
for the Hierarchical Supervisor (Protein-Critic) training.
"""

import json
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def build_multitask_dataset():
    out_dir = Path("data/processed/protein_lm/multitask")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    samples = []
    
    # 1. Load UniProt Data
    print("[*] Loading UniProt protein sequences...")
    with open("data/processed/protein_pfam_labels.json", "r") as f:
        protein_seqs = json.load(f)
        
    print("[*] Loading UniProt metadata...")
    df_meta = pd.read_csv("data/processed/uniprot_metadata_full.csv")
    
    # De-duplicate metadata by ncbi_id (take the first match if multiple UniProt ACs)
    df_meta = df_meta.drop_duplicates(subset=["ncbi_id"])
    meta_dict = df_meta.set_index("ncbi_id").to_dict(orient="index")
    
    uniprot_added = 0
    for ncbi_id, seq_info in protein_seqs.items():
        meta = meta_dict.get(ncbi_id)
        if not meta: continue
        
        # Parse Pfam (take the first one if multiple are separated by ;)
        pfam_raw = meta.get("pfam")
        pfam = str(pfam_raw).split(";")[0].strip() if pd.notna(pfam_raw) and pfam_raw else None
        
        # Parse EC
        ec_raw = meta.get("ec")
        ec = str(ec_raw).split(";")[0].strip() if pd.notna(ec_raw) and ec_raw else None
        
        if pfam or ec:
            samples.append({
                "sequence": seq_info["sequence"],
                "pfam": pfam,
                "ec": ec,
                "stability": None,  # Missing
                "source": "uniprot"
            })
            uniprot_added += 1
            
    print(f"[*] Added {uniprot_added} UniProt samples with Pfam or EC labels.")
    
    # 2. Load MegaScale Stability Data
    print("[*] Loading MegaScale stability data...")
    stability_files = [
        "data/processed/protein_lm/stability/stability_train.jsonl",
        "data/processed/protein_lm/stability/stability_val.jsonl"
    ]
    
    stability_added = 0
    for file_path in stability_files:
        path = Path(file_path)
        if not path.exists(): continue
        with open(path, "r") as f:
            for line in f:
                data = json.loads(line)
                samples.append({
                    "sequence": data["sequence"],
                    "pfam": None,  # Missing
                    "ec": None,    # Missing
                    "stability": data["stability_label"],
                    "source": "megascale"
                })
                stability_added += 1
                
    print(f"[*] Added {stability_added} MegaScale samples.")
    
    # 3. Create Label Vocabularies (Top N to avoid rare classes)
    print("[*] Building label vocabularies...")
    df_samples = pd.DataFrame(samples)
    
    # Keep top 1000 Pfams
    top_pfams = df_samples['pfam'].value_counts().head(1000).index.tolist()
    pfam_vocab = {p: i for i, p in enumerate(top_pfams)}
    
    # Keep top 500 ECs
    top_ecs = df_samples['ec'].value_counts().head(500).index.tolist()
    ec_vocab = {e: i for i, e in enumerate(top_ecs)}
    
    # Stability (Binary)
    stability_vocab = {"stable": 0, "unstable": 1}
    
    # Save vocabularies
    vocabs = {
        "pfam": pfam_vocab,
        "ec": ec_vocab,
        "stability": stability_vocab
    }
    with open(out_dir / "task_vocabs.json", "w") as f:
        json.dump(vocabs, f, indent=4)
        
    # 4. Map labels to integers and split
    print("[*] Mapping labels and splitting dataset...")
    final_samples = []
    for s in samples:
        final_samples.append({
            "sequence": s["sequence"],
            "pfam_id": pfam_vocab.get(s["pfam"], -1) if s["pfam"] else -1,
            "ec_id": ec_vocab.get(s["ec"], -1) if s["ec"] else -1,
            "stability_id": stability_vocab.get(s["stability"], -1) if s["stability"] else -1,
        })
        
    train, val = train_test_split(final_samples, test_size=0.1, random_state=42)
    
    with open(out_dir / "train.jsonl", "w") as f:
        for s in train: f.write(json.dumps(s) + "\n")
        
    with open(out_dir / "val.jsonl", "w") as f:
        for s in val: f.write(json.dumps(s) + "\n")
        
    print(f"[success] Saved {len(train)} train and {len(val)} val samples to {out_dir}")

if __name__ == "__main__":
    build_multitask_dataset()
