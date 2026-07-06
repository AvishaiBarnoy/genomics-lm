import random
import numpy as np

def pairwise_identity(seqs: list[str], max_pairs: int = 500) -> float:
    """Average pairwise sequence identity (fraction of identical positions)."""
    if len(seqs) < 2:
        return 1.0
    pairs = [(seqs[i], seqs[j]) for i in range(len(seqs)) for j in range(i+1, len(seqs))]
    if len(pairs) > max_pairs:
        pairs = random.sample(pairs, max_pairs)
    identities = []
    for a, b in pairs:
        min_len = min(len(a), len(b))
        if min_len == 0:
            continue
        matches = sum(x == y for x, y in zip(a[:min_len], b[:min_len]))
        identities.append(matches / min_len)
    return float(np.mean(identities)) if identities else 0.0


def kmer_diversity(seqs: list[str], k: int = 3) -> float:
    """Fraction of possible k-mers observed across all sequences (normalised)."""
    observed = set()
    for seq in seqs:
        for i in range(len(seq) - k + 1):
            observed.add(seq[i:i+k])
    total_possible = 20 ** k  # amino acid k-mers
    return len(observed) / total_possible


def gc_content(codon_seqs: list[list[str]]) -> list[float]:
    """GC content per sequence."""
    results = []
    for codons in codon_seqs:
        dna = "".join(codons)
        if not dna:
            results.append(0.0)
            continue
        gc = sum(1 for c in dna.upper() if c in "GC")
        results.append(gc / len(dna))
    return results
