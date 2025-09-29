#!/usr/bin/env python3
"""
Convert CDS DNA (one sequence per line) → sequences of codon ids.
Vocabulary = 64 codons + specials: <pad>, <bos>, <eos>, <eog> (end-of-gene), <unk>

Outputs:
- data/processed/codon_ids.txt  (space-separated ints, one CDS per line)
- data/processed/vocab_codon.txt (id -> token)
"""

from pathlib import Path
import argparse

CODONS = [a+b+c for a in "ACGT" for b in "ACGT" for c in "ACGT"]
SPECIALS = ["<pad>", "<bos>", "<eos>", "<eog>", "<unk>"]
VOCAB = SPECIALS + CODONS
stoi = {tok:i for i,tok in enumerate(VOCAB)}
itos = {i:t for t,i in stoi.items()}

def to_ids(dna: str) -> list[int]:
    dna = dna.strip().upper().replace("U","T")
    if len(dna) < 3: return []
    # trim to a multiple of 3, left-aligned (GenBank CDS already in-frame)
    L = len(dna) // 3 * 3
    ids = [stoi["<bos>"]]
    unk = stoi["<unk>"]
    for i in range(0, L, 3):
        codon = dna[i:i+3]
        ids.append(stoi.get(codon, unk))
    ids.append(stoi["<eog>"])
    return ids

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="data/processed/cds_dna.txt")
    ap.add_argument("--out_ids", default="data/processed/codon_ids.txt")
    ap.add_argument("--out_vocab", default="data/processed/vocab_codon.txt")
    args = ap.parse_args()

    ids_path = Path(args.out_ids); ids_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.inp) as fin, open(args.out_ids, "w") as fout:
        n=0
        for line in fin:
            arr = to_ids(line)
            if len(arr)>2:
                fout.write(" ".join(map(str, arr))+"\n")
                n+=1
    with open(args.out_vocab, "w") as f:
        for i,tok in enumerate(VOCAB):
            f.write(f"{i}\t{tok}\n")
    print(f"[tokenize] wrote {n} sequences → {ids_path} | vocab size {len(VOCAB)}")

if __name__ == "__main__":
    main()

