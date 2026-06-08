#!/usr/bin/env python3
"""
Extracts contiguous "Genomic Tapes" (chromosomal segments) from GenBank files.
Moves beyond single-CDS extraction to capture operon logic and intergenic regions.
"""

import argparse
from pathlib import Path
from Bio import SeqIO
import numpy as np

def extract_tapes(gb_path, window_bp, stride_bp, out_f, out_m, genome_id):
    """
    Slides a window across the chromosome and writes the DNA tape.
    """
    idx = 0
    for rec in SeqIO.parse(gb_path, "genbank"):
        seq = str(rec.seq).upper()
        L = len(seq)
        
        # Only process forward strand for now (Genomic context is strand-less until translation)
        # But we align to the reference strand.
        for start in range(0, L - window_bp + 1, stride_bp):
            end = start + window_bp
            tape_dna = seq[start:end]
            
            # Filter for pure ACGT
            if set(tape_dna) <= set("ACGT"):
                out_f.write(tape_dna + "\n")
                # Metadata: idx, genome, start_pos, end_pos
                out_m.write(f"{idx}\t{genome_id}\t{start}\t{end}\n")
                idx += 1
    return idx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gbff", nargs="+", required=True, help="Input GenBank files")
    ap.add_argument("--window_bp", type=int, default=1536, help="Window size in BP (512 codons)")
    ap.add_argument("--stride_bp", type=int, default=768, help="Stride in BP (256 codons)")
    ap.add_argument("--out_txt", default="data/processed/genomic_tape.txt")
    ap.add_argument("--out_meta", default="data/processed/genomic_tape_meta.tsv")
    args = ap.parse_args()

    Path(args.out_txt).parent.mkdir(parents=True, exist_ok=True)
    
    total_segments = 0
    with open(args.out_txt, "w") as ft, open(args.out_meta, "w") as fm:
        fm.write("line_idx\tgenome\tstart\tend\n")
        for gb in args.gbff:
            # Better genome ID extraction (GCF_XXXXXXXXX)
            parts = Path(gb).stem.split("_")
            genome_id = "_".join(parts[:2]) if len(parts) >= 2 else parts[0]
            
            print(f"[*] Processing {genome_id}...")
            count = extract_tapes(gb, args.window_bp, args.stride_bp, ft, fm, genome_id)
            total_segments += count
            
    print(f"[success] Extracted {total_segments} genomic tape segments.")

if __name__ == "__main__":
    main()
