#!/usr/bin/env python3
"""
Writes:
- data/processed/cds_dna.txt   (one CDS per line)
- data/processed/cds_meta.tsv  (parallel, cols: line_idx, genome_id)
"""

from pathlib import Path
import argparse
from Bio import SeqIO

def reverse_complement(s):
    comp = str.maketrans("ACGTacgtnN","TGCAtgcann")
    return s.translate(comp)[::-1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gbff", nargs="+", required=True)
    ap.add_argument("--out_txt", default="data/processed/cds_dna.txt")
    ap.add_argument("--out_meta", default="data/processed/cds_meta.tsv")
    ap.add_argument("--min_len", type=int, default=90)
    args = ap.parse_args()

    Path(args.out_txt).parent.mkdir(parents=True, exist_ok=True)
    idx = 0
    with open(args.out_txt,"w") as ft, open(args.out_meta,"w") as fm:
        fm.write("line_idx\tgenome\n")
        for gb in args.gbff:
            genome_id = Path(gb).stem.split("_")[0]
            for rec in SeqIO.parse(gb, "genbank"):
                seq = str(rec.seq).upper()
                for feat in rec.features:
                    if feat.type!="CDS": continue
                    s,e = int(feat.location.start), int(feat.location.end)
                    strand = int(feat.location.strand or 1)
                    cds = seq[s:e]
                    if strand==-1: cds = reverse_complement(cds)
                    if len(cds) >= args.min_len and set(cds) <= set("ACGTN"):
                        ft.write(cds+"\n")
                        fm.write(f"{idx}\t{genome_id}\n")
                        idx+=1
    print(f"[extract] wrote {idx} CDS with meta.")
if __name__=="__main__": main()
