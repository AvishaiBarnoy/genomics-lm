#!/usr/bin/env python3
"""
Prepares the Enzyme Commission (EC) level-1 classification dataset.
Parses all gbff files under data/raw/ recursively, matches protein_id qualifiers
to uniprot_metadata_full.csv annotations, and writes a split train/test dataset.
"""

from pathlib import Path
import pandas as pd
from Bio import SeqIO
from sklearn.model_selection import train_test_split

def reverse_complement(s):
    comp = str.maketrans("ACGTacgtnN", "TGCAtgcann")
    return s.translate(comp)[::-1]

def main():
    repo_root = Path(__file__).resolve().parents[1]
    raw_dir = repo_root / "data" / "raw"
    processed_dir = repo_root / "data" / "processed"
    labels_dir = repo_root / "data" / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load UniProt metadata mapping
    metadata_path = processed_dir / "uniprot_metadata_full.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"UniProt metadata not found at {metadata_path}")

    print("[*] Loading UniProt metadata...")
    df_meta = pd.read_csv(metadata_path)
    # Filter for entries with valid EC annotations
    df_meta = df_meta.dropna(subset=["ec"])
    # Map ncbi_id to the first digit of the EC number (EC Class 1-7)
    ec_map = {}
    for _, row in df_meta.iterrows():
        ncbi_id = str(row["ncbi_id"]).strip()
        ec_str = str(row["ec"]).strip()
        if ec_str and ec_str[0].isdigit():
            class_digit = int(ec_str[0])
            if 1 <= class_digit <= 7:
                ec_map[ncbi_id] = class_digit

    print(f"[+] Loaded {len(ec_map)} unique protein ID mappings to EC Class 1-7.")

    # 2. Scan GenBank files recursively
    gbff_files = list(raw_dir.glob("**/*.gbff"))
    print(f"[*] Found {len(gbff_files)} GenBank (.gbff) files to scan.")

    samples = []
    seen_ids = set()

    for gb_path in gbff_files:
        print(f"[*] Parsing: {gb_path.name}")
        for rec in SeqIO.parse(gb_path, "genbank"):
            seq = str(rec.seq).upper()
            for feat in rec.features:
                if feat.type != "CDS":
                    continue

                # Extract protein RefSeq accession
                if "protein_id" not in feat.qualifiers:
                    continue
                pid = feat.qualifiers["protein_id"][0]

                if pid in seen_ids:
                    continue

                if pid in ec_map:
                    # Extract CDS nucleotide sequence
                    s, e = int(feat.location.start), int(feat.location.end)
                    strand = int(feat.location.strand or 1)
                    cds_seq = seq[s:e]
                    if strand == -1:
                        cds_seq = reverse_complement(cds_seq)

                    # Ensure valid sequence
                    if len(cds_seq) >= 90 and set(cds_seq) <= set("ACGTN"):
                        samples.append({
                            "id": pid,
                            "seq": cds_seq,
                            "label": ec_map[pid]
                        })
                        seen_ids.add(pid)

    print(f"[+] Extracted {len(samples)} matched sequences with EC labels.")
    if len(samples) == 0:
        print("[!] No matching sequences found. Exiting.")
        return

    # 3. Create datasets and split
    df_dataset = pd.DataFrame(samples)

    # Save master dataset
    dataset_csv = processed_dir / "ec_sequences_labeled.csv"
    df_dataset.to_csv(dataset_csv, index=False)
    print(f"[+] Wrote master dataset to {dataset_csv}")

    # Split 80/20 train/test stratified by label
    train_df, test_df = train_test_split(
        df_dataset, test_size=0.2, random_state=42, stratify=df_dataset["label"]
    )

    # Save labels files as expected by train_classifier.py
    train_labels_csv = labels_dir / "train_ec.csv"
    test_labels_csv = labels_dir / "test_ec.csv"

    train_df[["id", "label"]].to_csv(train_labels_csv, index=False)
    test_df[["id", "label"]].to_csv(test_labels_csv, index=False)

    # Save sequences FASTA/CSV for embedding extraction
    train_seqs_csv = processed_dir / "ec_train_seqs.csv"
    test_seqs_csv = processed_dir / "ec_test_seqs.csv"
    train_df[["id", "seq"]].to_csv(train_seqs_csv, index=False)
    test_df[["id", "seq"]].to_csv(test_seqs_csv, index=False)

    print(f"[success] Created EC classification splits:")
    print(f"  - Train: {len(train_df)} samples -> {train_labels_csv} & {train_seqs_csv}")
    print(f"  - Test:  {len(test_df)} samples -> {test_labels_csv} & {test_seqs_csv}")

if __name__ == "__main__":
    main()
