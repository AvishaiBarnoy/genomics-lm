#!/usr/bin/env python3
"""
Reads hybrid sequence TSV data, tokenizes them using HybridTokenizer,
and writes out space-separated token IDs and vocabulary files.
"""

import argparse
from pathlib import Path
from src.codonlm.hybrid_tokenizer import HybridTokenizer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="data/processed/hybrid_data.tsv")
    ap.add_argument("--out_ids", default="data/processed/hybrid_ids.txt")
    ap.add_argument("--out_vocab", default="data/processed/vocab_hybrid.txt")
    ap.add_argument("--out_itos", default="data/processed/itos_hybrid.txt")
    args = ap.parse_args()

    ids_path = Path(args.out_ids)
    ids_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = HybridTokenizer()

    n = 0
    with open(args.inp) as fin, open(args.out_ids, "w") as fout:
        # Read header
        header = next(fin).strip().split("\t")
        col_to_idx = {col: i for i, col in enumerate(header)}

        for line in fin:
            parts = line.strip().split("\t")
            if len(parts) < len(header):
                continue
            
            seq = parts[col_to_idx["sequence"]]
            cds_start = int(parts[col_to_idx["cds_start"]])
            cds_end = int(parts[col_to_idx["cds_end"]])

            # Encode using the HybridTokenizer
            # The strand is always '+' relative to the extracted transcription-oriented sequence.
            token_ids = tokenizer.encode(seq, [(cds_start, cds_end, "+")])
            
            if token_ids:
                fout.write(" ".join(map(str, token_ids)) + "\n")
                n += 1

    # Write vocab
    with open(args.out_vocab, "w") as f:
        for idx, tok in enumerate(tokenizer.vocab):
            f.write(f"{idx}\t{tok}\n")

    # Write itos
    with open(args.out_itos, "w") as f:
        for tok in tokenizer.vocab:
            f.write(f"{tok}\n")

    print(f"[hybrid_tokenize] Wrote {n} tokenized sequences → {ids_path} | vocab size {tokenizer.vocab_size}")

if __name__ == "__main__":
    main()
