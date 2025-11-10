#!/usr/bin/env python3
"""
Convert CDS DNA (one sequence per line) → sequences of codon ids.

Vocabulary (fixed order):
  0: <PAD>
  1: <BOS_CDS>
  2: <EOS_CDS>
  3: <SEP>
  4..67: the 64 codons (AAA..TTT in lexical order)

Encoding a single CDS yields:
  ["<BOS_CDS>", CODON1, ..., STOPCODON, "<EOS_CDS>"]

When packing multiple CDS into one sequence, separate by <SEP>. We include
<EOS_CDS> before every <SEP> so models can learn explicit termination.

Outputs:
- data/processed/codon_ids.txt  (space-separated ints, one CDS per line)
- data/processed/vocab_codon.txt (id -> token)
- data/processed/itos_codon.txt  (token per line)
"""

from pathlib import Path
import argparse

CODONS = [a + b + c for a in "ACGT" for b in "ACGT" for c in "ACGT"]
STOP_CODONS = {"TAA", "TAG", "TGA"}
SPECIALS = ["<PAD>", "<BOS_CDS>", "<EOS_CDS>", "<SEP>"]
VOCAB = SPECIALS + CODONS
stoi = {tok: i for i, tok in enumerate(VOCAB)}
# itos must map to canonical tokens; build from VOCAB only
itos = {i: tok for i, tok in enumerate(VOCAB)}
# Backward-compat aliases for legacy tests/configs (affect stoi only)
ALIASES = {
    "<bos>": "<BOS_CDS>",
    "<eog>": "<EOS_CDS>",
    "<eos>": "<EOS_CDS>",
}
for alias, canonical in ALIASES.items():
    stoi[alias] = stoi[canonical]

def to_ids(dna: str) -> list[int]:
    dna = dna.strip().upper().replace("U", "T")
    if len(dna) < 3:
        return []
    # trim to a multiple of 3, left-aligned (GenBank CDS already in-frame)
    L = (len(dna) // 3) * 3
    ids = [stoi["<BOS_CDS>"]]
    for i in range(0, L, 3):
        codon = dna[i : i + 3]
        idx = stoi.get(codon)
        if idx is None:
            # Skip unknown codons (should not happen for ACGT)
            continue
        ids.append(idx)
    # Ensure we terminate with EOS_CDS even if sequence lacked a canonical stop
    if len(ids) == 1:
        return []
    if itos.get(ids[-1]) not in STOP_CODONS:
        # do nothing; keep whatever last codon was (ground truth may not end with stop in some corpora)
        pass
    ids.append(stoi["<EOS_CDS>"])
    return ids

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="data/processed/cds_dna.txt")
    ap.add_argument("--out_ids", default="data/processed/codon_ids.txt")
    ap.add_argument("--out_vocab", default="data/processed/vocab_codon.txt")
    ap.add_argument("--out_itos", default="data/processed/itos_codon.txt")
    args = ap.parse_args()

    ids_path = Path(args.out_ids); ids_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.inp) as fin, open(args.out_ids, "w") as fout:
        n=0
        for line in fin:
            arr = to_ids(line)
            if len(arr) > 2:
                fout.write(" ".join(map(str, arr)) + "\n")
                n += 1
    with open(args.out_vocab, "w") as f:
        for i, tok in enumerate(VOCAB):
            f.write(f"{i}\t{tok}\n")
    with open(args.out_itos, "w") as f:
        for tok in VOCAB:
            f.write(f"{tok}\n")
    print(
        f"[tokenize] wrote {n} sequences → {ids_path} | vocab size {len(VOCAB)} | itos {args.out_itos}"
    )

__all__ = ["CODONS", "STOP_CODONS", "VOCAB", "stoi", "itos", "to_ids"]

if __name__ == "__main__":
    main()
