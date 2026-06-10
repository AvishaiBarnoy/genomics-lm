#!/usr/bin/env python3
"""
Prepares or generates mock/synthetic benchmark datasets for SOTA prokaryotic evaluation.
This ensures a self-contained environment for running the benchmark scripts.

Datasets generated:
1. data/benchmarks/protein_dms.csv (Zero-Shot Protein DMS)
2. data/benchmarks/rrna_dms.csv (E. coli 5S rRNA DMS)
3. data/benchmarks/kosuri_promoters.csv (Kosuri promoter/RBS expression)
4. data/benchmarks/lambda_essentiality.csv (Lambda phage gene essentiality)
5. data/benchmarks/pseudomonas_essentiality.csv (P. aeruginosa gene essentiality)
"""

import os
import random
import csv
from pathlib import Path

# Set seeds for reproducibility
random.seed(42)

def generate_random_dna(length):
    return "".join(random.choice("ACGT") for _ in range(length))

def generate_random_aa(length):
    AAS = "ARNDCQEGHILKMFPSTWYV"
    return "".join(random.choice(AAS) for _ in range(length))

def mutate_sequence(seq, pos, new_char):
    return seq[:pos] + new_char + seq[pos+1:]

def back_translate(aa_seq):
    # Map each amino acid to a codon (using standard bacterial codon usage representation)
    codon_map = {
        'A': 'GCG', 'R': 'CGT', 'N': 'AAC', 'D': 'GAT', 'C': 'TGC',
        'Q': 'CAG', 'E': 'GAA', 'G': 'GGC', 'H': 'CAT', 'I': 'ATC',
        'L': 'CTG', 'K': 'AAA', 'M': 'ATG', 'F': 'TTC', 'P': 'CCG',
        'S': 'TCC', 'T': 'ACC', 'W': 'TGG', 'Y': 'TAC', 'V': 'GTG'
    }
    return "".join(codon_map.get(aa, 'GCG') for aa in aa_seq)

def main():
    dest_dir = Path("data/benchmarks")
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Creating benchmark datasets in {dest_dir}...")

    # 1. Protein DMS (prokaryotic protein)
    # Wild-type AA sequence
    wt_protein = "MREIVLHQAGQCGNQIGAKFWEVISDEHGIDPTGTYHGDSDLQLERINVYFNEATGGRYV"
    wt_dna = back_translate(wt_protein)
    
    protein_dms_rows = []
    # Generate single amino acid mutations
    AAS = "ARNDCQEGHILKMFPSTWYV"
    for pos in range(len(wt_protein)):
        wt_aa = wt_protein[pos]
        # Choose 3 mutant amino acids per position
        mutants = [a for a in AAS if a != wt_aa]
        random.shuffle(mutants)
        for mut_aa in mutants[:3]:
            mut_protein = mutate_sequence(wt_protein, pos, mut_aa)
            mut_dna = back_translate(mut_protein)
            # Fitness score is modeled: negative for drastic changes, small for similar
            # We mock it to have some correlation with sequence differences
            fitness = -abs(ord(wt_aa) - ord(mut_aa)) / 10.0 + random.normalvariate(0, 0.5)
            protein_dms_rows.append({
                "wildtype_seq": wt_dna,
                "mutated_seq": mut_dna,
                "wildtype_aa": wt_protein,
                "mutated_aa": mut_protein,
                "mutation": f"{wt_aa}{pos+1}{mut_aa}",
                "fitness_score": f"{fitness:.4f}"
            })
            
    with open(dest_dir / "protein_dms.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["wildtype_seq", "mutated_seq", "wildtype_aa", "mutated_aa", "mutation", "fitness_score"])
        writer.writeheader()
        writer.writerows(protein_dms_rows)
    print(f"  - Wrote protein_dms.csv ({len(protein_dms_rows)} variants)")

    # 2. E. coli 5S rRNA DMS
    # 5S rRNA wt sequence (120 nt)
    wt_rrna = "TCTTGACGGCGACCATTGCAAATTGCTCGTCAGCTCAGTGGGATTGGCAGCCTGGCTTGACATGGTGCAATCGGGTCATGAGCGTGCACGATCGTCTTTG"
    rrna_dms_rows = []
    for pos in range(len(wt_rrna)):
        wt_nt = wt_rrna[pos]
        mut_nts = [n for n in "ACGT" if n != wt_nt]
        for mut_nt in mut_nts:
            mut_rrna = mutate_sequence(wt_rrna, pos, mut_nt)
            # Heuristic fitness: mock experimental rank
            fitness = -0.5 if pos % 5 == 0 else -0.1 + random.normalvariate(0, 0.2)
            rrna_dms_rows.append({
                "wildtype_seq": wt_rrna,
                "mutated_seq": mut_rrna,
                "mutation": f"{wt_nt}{pos+1}{mut_nt}",
                "fitness_score": f"{fitness:.4f}"
            })
            
    with open(dest_dir / "rrna_dms.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["wildtype_seq", "mutated_seq", "mutation", "fitness_score"])
        writer.writeheader()
        writer.writerows(rrna_dms_rows)
    print(f"  - Wrote rrna_dms.csv ({len(rrna_dms_rows)} variants)")

    # 3. Kosuri Promoter/RBS Library
    # 100 sequences with different expression scores
    kosuri_rows = []
    for i in range(150):
        # Generate random DNA sequence (75-90 nt)
        seq_len = random.randint(75, 90)
        seq = generate_random_dna(seq_len)
        # Mock expression score: let's correlate with "TTGACA" (canonical -35 hexamer) or "TATAAT" (Pribnow box) presence
        score = 0.0
        if "TATA" in seq:
            score += 2.0
        if "TTG" in seq:
            score += 1.5
        score += random.normalvariate(0, 1.0)
        kosuri_rows.append({
            "sequence": seq,
            "expression_score": f"{score:.4f}"
        })
        
    with open(dest_dir / "kosuri_promoters.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sequence", "expression_score"])
        writer.writeheader()
        writer.writerows(kosuri_rows)
    print(f"  - Wrote kosuri_promoters.csv ({len(kosuri_rows)} promoters)")

    # 4. Lambda Phage Gene Essentiality
    lambda_rows = []
    for i in range(120):
        # Coding sequence of length 300 (100 codons)
        seq = generate_random_dna(300)
        # Force start codon ATG and stop codon TAA
        seq = "ATG" + seq[3:297] + "TAA"
        # Mock essential: 0 or 1
        essential = 1 if (i % 3 == 0 or "ATG" in seq[3:60]) else 0
        lambda_rows.append({
            "sequence": seq,
            "essential": essential
        })
        
    with open(dest_dir / "lambda_essentiality.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sequence", "essential"])
        writer.writeheader()
        writer.writerows(lambda_rows)
    print(f"  - Wrote lambda_essentiality.csv ({len(lambda_rows)} genes)")

    # 5. P. aeruginosa Gene Essentiality
    pseudomonas_rows = []
    for i in range(150):
        seq = generate_random_dna(300)
        seq = "ATG" + seq[3:297] + "TAA"
        essential = 1 if (i % 4 == 0 or "GGC" in seq[15:45]) else 0
        pseudomonas_rows.append({
            "sequence": seq,
            "essential": essential
        })
        
    with open(dest_dir / "pseudomonas_essentiality.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sequence", "essential"])
        writer.writeheader()
        writer.writerows(pseudomonas_rows)
    print(f"  - Wrote pseudomonas_essentiality.csv ({len(pseudomonas_rows)} genes)")
    
    print("[*] Benchmark datasets prepared successfully!")

if __name__ == "__main__":
    main()
