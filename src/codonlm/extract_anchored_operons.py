#!/usr/bin/env python3
"""
Extracts "Anchored Operon Bridges" from GenBank files.
A bridge is a fixed-length window centered on the boundary between two adjacent genes 
on the same strand (Stop of Gene A -> Intergenic -> Start of Gene B).
"""

import argparse
from pathlib import Path
from Bio import SeqIO
from Bio.Seq import Seq

def extract_anchored_bridges(gb_path, window_bp, out_f, out_m, genome_id):
    """
    Identifies adjacent genes on the same strand and extracts a centered window.
    """
    idx = 0
    half_win = window_bp // 2
    
    for rec in SeqIO.parse(gb_path, "genbank"):
        seq_full = str(rec.seq).upper()
        L = len(seq_full)
        
        # 1. Collect all CDS features with a valid strand
        cds_features = [f for f in rec.features if f.type == "CDS" and f.location.strand is not None]
        # Sort by start position
        cds_features.sort(key=lambda x: int(x.location.start))
        
        # 2. Find adjacent pairs on the same strand
        for i in range(len(cds_features) - 1):
            f1 = cds_features[i]
            f2 = cds_features[i+1]
            
            # Must be same strand (+1 or -1)
            if f1.location.strand != f2.location.strand:
                continue
            
            # Boundary calculation
            if f1.location.strand == 1:
                # midpoint is roughly between end of f1 and start of f2
                midpoint = (int(f1.location.end) + int(f2.location.start)) // 2
            else:
                midpoint = (int(f1.location.start) + int(f2.location.end)) // 2
                
            start = midpoint - half_win
            end = midpoint + half_win
            
            # Check bounds and Ns
            if start < 0 or end > L:
                continue
                
            bridge_dna = seq_full[start:end]
            
            if set(bridge_dna) <= set("ACGT"):
                # If reverse strand, take reverse complement to keep it sense-strand relative
                if f1.location.strand == -1:
                    bridge_dna = str(Seq(bridge_dna).reverse_complement())
                
                out_f.write(bridge_dna + "\n")
                # Metadata: idx, genome, midpoint, strand, gene1_id, gene2_id
                g1 = f1.qualifiers.get("locus_tag", ["unk"])[0]
                g2 = f2.qualifiers.get("locus_tag", ["unk"])[0]
                out_m.write(f"{idx}\t{genome_id}\t{midpoint}\t{f1.location.strand}\t{g1}\t{g2}\n")
                idx += 1
                
    return idx

def main():
    """Extracts anchored operon bridges from GenBank files and writes them to text and metadata files."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--gbff", nargs="+", required=True)
    ap.add_argument("--window_bp", type=int, default=1536, help="Window size in BP (512 codons)")
    ap.add_argument("--out_txt", default="data/processed/operon_bridges.txt")
    ap.add_argument("--out_meta", default="data/processed/operon_bridges_meta.tsv")
    args = ap.parse_args()

    Path(args.out_txt).parent.mkdir(parents=True, exist_ok=True)
    
    total_bridges = 0
    with open(args.out_txt, "w") as ft, open(args.out_meta, "w") as fm:
        fm.write("line_idx\tgenome\tmidpoint\tstrand\tgene1\tgene2\n")
        for gb in args.gbff:
            parts = Path(gb).stem.split("_")
            genome_id = "_".join(parts[:2]) if len(parts) >= 2 else parts[0]
            
            print(f"[*] Extracting bridges from {genome_id}...")
            count = extract_anchored_bridges(gb, args.window_bp, ft, fm, genome_id)
            total_bridges += count
            
    print(f"[success] Extracted {total_bridges} anchored operon bridges.")


if __name__ == "__main__":
    main()
