#!/usr/bin/env python3
"""
Extract CDS nucleotide sequences from one or more GenBank .gbff files.
Writes a FASTA-like plain text with one CDS per line (no headers) to data/processed/cds_dna.txt

Why:
- Keep only coding regions (frame is known).
- Simplifies tokenization to in-frame codons later.

Key parameters:
- --min_len: discard very short CDS (noisy / fragments).
"""

from pathlib import Path
import argparse
from Bio import SeqIO

def reverse_complement(s: str) -> str:
    comp = str.maketrans("ACGTacgtnN", "TGCAtgcann")
    return s.translate(comp)[::-1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gbff", nargs="+", required=True, help="GenBank files")
    ap.add_argument("--out", default="data/processed/cds_dna.txt")
    ap.add_argument("--min_len", type=int, default=90, help="min CDS length (bp)")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    kept, skipped = 0, 0
    with out.open("w") as f:
        for gb in args.gbff:
            for rec in SeqIO.parse(gb, "genbank"):
                genome = str(rec.seq).upper()
                for feat in rec.features:
                    if feat.type != "CDS": 
                        continue
                    try:
                        start = int(feat.location.start)
                        end   = int(feat.location.end)
                        strand = int(feat.location.strand or 1)
                        seq = genome[start:end]
                        if strand == -1:
                            seq = reverse_complement(seq)
                        if len(seq) >= args.min_len and set(seq) <= set("ACGTN"):
                            f.write(seq + "\n")
                            kept += 1
                        else:
                            skipped += 1
                    except Exception:
                        skipped += 1
    print(f"[extract] wrote {kept} CDS; skipped {skipped}; → {out}")

if __name__ == "__main__":
    main()

