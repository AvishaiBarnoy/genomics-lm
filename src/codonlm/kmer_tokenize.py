#!/usr/bin/env python3
"""
General k-mer tokenizer for DNA (not frame-dependent).
Args:
  --k 6
Outputs:
  data/processed/kmer_ids.txt
  data/processed/vocab_kmer.txt
"""
from itertools import product
import argparse

def build_vocab(k):
    alphabet = "ACGT"
    toks = ["".join(p) for p in product(alphabet, repeat=k)]
    specials = ["<pad>","<bos>","<eos>","<unk>"]
    return specials + toks

def to_ids(seq, k, stoi):
    s = seq.strip().upper().replace("U","T")
    ids = [stoi["<bos>"]]
    for i in range(0, len(s)-k+1):
        ids.append(stoi.get(s[i:i+k], stoi["<unk>"]))
    ids.append(stoi["<eos>"])
    return ids

# (implementation similar to codon_tokenize.py; omitted for brevity)

