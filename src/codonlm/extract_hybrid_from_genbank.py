#!/usr/bin/env python3
"""
Extracts hybrid sequences (30 bp upstream + CDS + 60 bp downstream) from GenBank files.
Saves sequence, oriented CDS boundaries (start, end), and genome metadata.
"""

import argparse
from pathlib import Path
from Bio import SeqIO

def reverse_complement(s: str) -> str:
    """Computes the reverse complement of a nucleotide sequence."""
    comp = str.maketrans("ACGTacgtnN", "TGCAtgcann")
    return s.translate(comp)[::-1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gbff", nargs="+", required=True)
    ap.add_argument("--out_tsv", default="data/processed/hybrid_data.tsv")
    ap.add_argument("--out_meta", default="data/processed/hybrid_meta.tsv")
    ap.add_argument("--min_len", type=int, default=90)
    ap.add_argument("--upstream", type=int, default=30)
    ap.add_argument("--downstream", type=int, default=60)
    args = ap.parse_args()

    Path(args.out_tsv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_meta).parent.mkdir(parents=True, exist_ok=True)
    idx = 0

    with open(args.out_tsv, "w") as f, open(args.out_meta, "w") as fm:
        # Header
        f.write("line_idx\tgenome\tsequence\tcds_start\tcds_end\n")
        fm.write("line_idx\tgenome\n")

        for gb in args.gbff:
            parts = Path(gb).stem.split("_")
            if len(parts) >= 2:
                genome_id = "_".join(parts[:2])
            else:
                genome_id = parts[0]

            for rec in SeqIO.parse(gb, "genbank"):
                seq = str(rec.seq).upper()
                seq_len = len(seq)

                for feat in rec.features:
                    if feat.type != "CDS":
                        continue

                    s, e = int(feat.location.start), int(feat.location.end)
                    strand = int(feat.location.strand or 1)
                    cds_len = e - s

                    if cds_len < args.min_len:
                        continue

                    if strand == 1:
                        # Forward strand: upstream is before s, downstream is after e
                        s_flank = max(0, s - args.upstream)
                        e_flank = min(seq_len, e + args.downstream)
                        
                        extracted = seq[s_flank:e_flank]
                        cds_start = s - s_flank
                        cds_end = e - s_flank
                    else:
                        # Reverse strand: upstream is after e, downstream is before s
                        # Genomic coordinates to extract: [s - downstream, e + upstream]
                        s_flank = max(0, s - args.downstream)
                        e_flank = min(seq_len, e + args.upstream)
                        
                        raw_region = seq[s_flank:e_flank]
                        extracted = reverse_complement(raw_region)
                        
                        # In reverse complement:
                        # The upstream region [e, e_flank] comes first
                        cds_start = e_flank - e
                        cds_end = e_flank - s

                    # Validate sequence characters
                    if set(extracted) <= set("ACGTN"):
                        f.write(f"{idx}\t{genome_id}\t{extracted}\t{cds_start}\t{cds_end}\n")
                        fm.write(f"{idx}\t{genome_id}\n")
                        idx += 1

    print(f"[extract_hybrid] Wrote {idx} hybrid records to {args.out_tsv} and meta to {args.out_meta}")

if __name__ == "__main__":
    main()
