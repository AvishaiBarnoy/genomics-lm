#!/usr/bin/env python3
"""
Writes:
- data/processed/cds_dna.txt   (one CDS per line)
- data/processed/cds_meta.tsv  (parallel CDS metadata)
"""

from pathlib import Path
import argparse
from Bio import SeqIO

def reverse_complement(s):
    """Computes the reverse complement of a nucleotide sequence."""
    comp = str.maketrans("ACGTacgtnN","TGCAtgcann")
    return s.translate(comp)[::-1]


def _first_qualifier(feat, key: str) -> str:
    values = feat.qualifiers.get(key, [])
    if not values:
        return ""
    return str(values[0])


def _join_qualifier(feat, key: str) -> str:
    values = feat.qualifiers.get(key, [])
    return ";".join(str(value) for value in values)

def main():
    """Extracts coding sequences (CDS) from GenBank files and writes them to text and metadata files."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--gbff", nargs="+", required=True)
    ap.add_argument("--out_txt", default="data/processed/cds_dna.txt")
    ap.add_argument("--out_meta", default="data/processed/cds_meta.tsv")
    ap.add_argument("--min_len", type=int, default=90)
    args = ap.parse_args()

    Path(args.out_txt).parent.mkdir(parents=True, exist_ok=True)
    idx = 0
    with open(args.out_txt,"w") as ft, open(args.out_meta,"w") as fm:
        fm.write(
            "line_idx\tgenome\trecord_id\tprotein_id\tlocus_tag\tgene\tproduct\t"
            "translation\tdb_xref\tstart\tend\tstrand\n"
        )
        for gb in args.gbff:
            # Capture GCF_000005845 or similar (first two parts)
            parts = Path(gb).stem.split("_")
            if len(parts) >= 2:
                genome_id = "_".join(parts[:2])
            else:
                genome_id = parts[0]
            
            for rec in SeqIO.parse(gb, "genbank"):
                seq = str(rec.seq).upper()
                for feat in rec.features:
                    if feat.type != "CDS":
                        continue
                    s,e = int(feat.location.start), int(feat.location.end)
                    strand = int(feat.location.strand or 1)
                    cds = seq[s:e]
                    if strand == -1:
                        cds = reverse_complement(cds)
                    if len(cds) >= args.min_len and set(cds) <= set("ACGTN"):
                        ft.write(cds+"\n")
                        meta = [
                            str(idx),
                            genome_id,
                            str(rec.id),
                            _first_qualifier(feat, "protein_id"),
                            _first_qualifier(feat, "locus_tag"),
                            _first_qualifier(feat, "gene"),
                            _first_qualifier(feat, "product"),
                            _first_qualifier(feat, "translation"),
                            _join_qualifier(feat, "db_xref"),
                            str(s),
                            str(e),
                            str(strand),
                        ]
                        fm.write("\t".join(value.replace("\t", " ") for value in meta) + "\n")
                        idx+=1
    print(f"[extract] wrote {idx} CDS with meta.")

if __name__ == "__main__":
    main()
