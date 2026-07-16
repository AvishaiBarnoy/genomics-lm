#!/usr/bin/env python3
"""
scripts/generate_synonymous_controls.py — Synonymous & Shuffling Control Suite Generator

This script generates control variants of a test dataset to scientifically validate
whether Genomics-LM encodes biological properties beyond basic composition and protein templates.

It outputs:
  1. Synonymous Mutated Control (preserves amino acid sequence, randomizes codon selection).
  2. Codon Shuffled Control (shuffles codon order within each gene, preserving composition).
  3. Protein Shuffled Control (shuffles amino acids, maintaining codon usage).

Usage:
  python -m scripts.generate_synonymous_controls --test_npz data/processed/global/my_run/test_bs256.npz
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
import numpy as np

# Standard Genetic Code
AA_TO_CODONS = {
    'A': ['GCT', 'GCC', 'GCA', 'GCG'],
    'R': ['CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
    'N': ['AAT', 'AAC'],
    'D': ['GAT', 'GAC'],
    'C': ['TGT', 'TGC'],
    'Q': ['CAA', 'CAG'],
    'E': ['GAA', 'GAG'],
    'G': ['GGT', 'GGC', 'GGA', 'GGG'],
    'H': ['CAT', 'CAC'],
    'I': ['ATT', 'ATC', 'ATA'],
    'L': ['TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG'],
    'K': ['AAA', 'AAG'],
    'M': ['ATG'],
    'F': ['TTT', 'TTC'],
    'P': ['CCT', 'CCC', 'CCA', 'CCG'],
    'S': ['TCT', 'TCC', 'TCA', 'TCG', 'AGT', 'AGC'],
    'T': ['ACT', 'ACC', 'ACA', 'ACG'],
    'W': ['TGG'],
    'Y': ['TAT', 'TAC'],
    'V': ['GTT', 'GTC', 'GTA', 'GTG'],
    '*': ['TAA', 'TAG', 'TGA']  # Stop codons
}

CODON_TO_AA = {codon: aa for aa, codons in AA_TO_CODONS.items() for codon in codons}

# Vocabulary mapping (must match codon_tokenize.py)
CODONS = [a + b + c for a in "ACGT" for b in "ACGT" for c in "ACGT"]
SPECIALS = ["<PAD>", "<BOS_CDS>", "<EOS_CDS>", "<SEP>"]
VOCAB = SPECIALS + CODONS
stoi = {tok: i for i, tok in enumerate(VOCAB)}
itos = {i: tok for i, tok in enumerate(VOCAB)}

def synonymous_mutate_sequence(seq: np.ndarray, rng: random.Random) -> np.ndarray:
    out = np.copy(seq)
    for i in range(len(seq)):
        val = int(seq[i])
        # Preserve specials (PAD=0, BOS=1, EOS=2, SEP=3)
        if val < 4:
            continue
        codon_str = itos[val]
        aa = CODON_TO_AA.get(codon_str)
        if aa is None:
            continue
        synonyms = AA_TO_CODONS[aa]
        chosen_codon = rng.choice(synonyms)
        out[i] = stoi[chosen_codon]
    return out

def codon_shuffle_sequence(seq: np.ndarray, rng: random.Random) -> np.ndarray:
    out = np.copy(seq)
    # Find positions of sense codons (ids >= 4)
    codon_indices = [i for i, val in enumerate(seq) if val >= 4]
    if not codon_indices:
        return out
    
    # Extract, shuffle, and re-insert
    codon_vals = [seq[i] for i in codon_indices]
    rng.shuffle(codon_vals)
    for idx, i in enumerate(codon_indices):
        out[i] = codon_vals[idx]
    return out

def protein_shuffle_sequence(seq: np.ndarray, rng: random.Random) -> np.ndarray:
    out = np.copy(seq)
    # Find indices of sense codons (ids >= 4)
    codon_indices = [i for i, val in enumerate(seq) if val >= 4]
    if not codon_indices:
        return out
        
    # Translate to amino acids
    aas = []
    for i in codon_indices:
        codon_str = itos[int(seq[i])]
        aas.append(CODON_TO_AA.get(codon_str, 'M'))  # Default to Methionine if not found
        
    # Shuffle amino acids
    rng.shuffle(aas)
    
    # Re-encode back to codon ids (randomly choosing synonymous codons for each shuffled amino acid)
    for idx, i in enumerate(codon_indices):
        aa = aas[idx]
        syns = AA_TO_CODONS[aa]
        chosen = rng.choice(syns)
        out[i] = stoi[chosen]
    return out

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_npz", required=True, help="Path to test NPZ dataset")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    test_path = Path(args.test_npz)
    if not test_path.exists():
        raise FileNotFoundError(f"Test NPZ not found: {test_path}")

    print(f"[controls] Loading test dataset from {test_path}...")
    with np.load(test_path) as data:
        X = data["X"]
        Y = data.get("Y", None)
        lengths = data.get("lengths", None)

    rng = random.Random(args.seed)

    # 1. Synonymous Recoding
    print("[controls] Generating Synonymous Recoding control...")
    X_syn = np.copy(X)
    for i in range(len(X)):
        X_syn[i] = synonymous_mutate_sequence(X[i], rng)
    
    Y_syn = None
    if Y is not None:
        Y_syn = np.copy(Y)
        for i in range(len(Y)):
            Y_syn[i] = synonymous_mutate_sequence(Y[i], rng)

    # 2. Codon Shuffling
    print("[controls] Generating Codon Shuffling control...")
    X_shuf = np.copy(X)
    for i in range(len(X)):
        X_shuf[i] = codon_shuffle_sequence(X[i], rng)
        
    Y_shuf = None
    if Y is not None:
        Y_shuf = np.copy(Y)
        for i in range(len(Y)):
            Y_shuf[i] = codon_shuffle_sequence(Y[i], rng)

    # 3. Protein Shuffling
    print("[controls] Generating Protein Shuffling control...")
    X_prot = np.copy(X)
    for i in range(len(X)):
        X_prot[i] = protein_shuffle_sequence(X[i], rng)
        
    Y_prot = None
    if Y is not None:
        Y_prot = np.copy(Y)
        for i in range(len(Y)):
            Y_prot[i] = protein_shuffle_sequence(Y[i], rng)

    # Save outputs
    out_dir = test_path.parent
    
    npz_syn_path = out_dir / f"test_control_synonymous_bs{X.shape[1] if X.ndim > 1 else 256}.npz"
    npz_shuf_path = out_dir / f"test_control_codon_shuffle_bs{X.shape[1] if X.ndim > 1 else 256}.npz"
    npz_prot_path = out_dir / f"test_control_protein_shuffle_bs{X.shape[1] if X.ndim > 1 else 256}.npz"

    def save_npz(path, x_arr, y_arr):
        if lengths is not None:
            np.savez_compressed(path, X=x_arr, lengths=lengths)
        else:
            np.savez_compressed(path, X=x_arr, Y=y_arr)
        print(f"[controls] Saved control dataset to {path}")

    save_npz(npz_syn_path, X_syn, Y_syn)
    save_npz(npz_shuf_path, X_shuf, Y_shuf)
    save_npz(npz_prot_path, X_prot, Y_prot)

if __name__ == "__main__":
    main()
