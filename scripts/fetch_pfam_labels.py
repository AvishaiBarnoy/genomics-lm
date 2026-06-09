#!/usr/bin/env python3
"""
Stage 3: Pfam Label Acquisition
Extracts protein IDs from GenBank files and fetches Pfam family labels via UniProt.
Provides the training labels for the Multi-Task Protein Classifier.
"""

import argparse
from pathlib import Path
from Bio import SeqIO
import json


def extract_protein_ids(gbff_files):
    """Parses GBFF files to map locus_tags to protein_ids and sequences."""
    mapping = {}
    print(f"[*] Parsing {len(gbff_files)} genome files...")
    for gb in gbff_files:
        for rec in SeqIO.parse(gb, "genbank"):
            for feat in rec.features:
                if feat.type == "CDS":
                    locus = feat.qualifiers.get("locus_tag", [None])[0]
                    prot_id = feat.qualifiers.get("protein_id", [None])[0]
                    seq = feat.qualifiers.get("translation", [None])[0]
                    if locus and prot_id and seq:
                        mapping[prot_id] = {
                            "locus_tag": locus,
                            "sequence": seq,
                            "genome": Path(gb).stem.split("_")[:2],
                        }
    return mapping


def fetch_pfam_batch(protein_ids):
    """
    Uses UniProt's ID mapping API to get Pfam labels.
    Note: Real implementation would handle pagination and large batches.
    For this prototype, we'll outline the API call logic.
    """
    # UniProt ID mapping API endpoint
    # This is a placeholder for the actual API interaction logic
    # In a real run, we would POST the IDs and wait for the mapping job to finish.
    print(f"[*] Simulating fetch for {len(protein_ids)} proteins...")
    # Mock result for logic flow
    return {pid: "PF00001" for pid in list(protein_ids)[:10]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--raw_dir", default="data/raw", help="Directory containing .gbff files"
    )
    ap.add_argument("--out", default="data/processed/protein_pfam_labels.json")
    args = ap.parse_args()

    gbff_files = list(Path(args.raw_dir).rglob("*.gbff"))
    if not gbff_files:
        print("[!] No .gbff files found.")
        return

    id_map = extract_protein_ids(gbff_files)
    print(f"[success] Extracted {len(id_map)} protein IDs.")

    # In a real Stage 3 session, we would run the full UniProt mapping here.
    # For now, we save the extracted protein metadata to be processed.
    with open(args.out, "w") as f:
        json.dump(id_map, f, indent=4)

    print(f"[save] Protein metadata saved to {args.out}")
    print("[*] Next Step: Use UniProt API to map these IDs to Pfam families.")


if __name__ == "__main__":
    main()
