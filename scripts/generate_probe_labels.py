#!/usr/bin/env python3
"""
Generate probe labels for codon tokens.

Reads the tokenizer vocabulary from ``runs/<run_id>/itos.txt`` and writes
``runs/<run_id>/probe_labels.csv`` with coarse biology-aware annotations used by
``scripts.probe_linear`` (Step 6 of the interpretability pipeline).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Tuple

RUNS_DIR = Path("runs")

# fmt: off
STANDARD_GENETIC_CODE: Dict[str, str] = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "TAT": "Y", "TAC": "Y", "TAA": "Stop", "TAG": "Stop",
    "TGT": "C", "TGC": "C", "TGA": "Stop", "TGG": "W",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}
# fmt: on

POLARITY_CLASS: Dict[str, str] = {
    # nonpolar (hydrophobic, uncharged)
    "A": "nonpolar",
    "V": "nonpolar",
    "L": "nonpolar",
    "I": "nonpolar",
    "M": "nonpolar",
    "F": "nonpolar",
    "W": "nonpolar",
    "P": "nonpolar",
    "G": "nonpolar",
    # polar (uncharged)
    "S": "polar",
    "T": "polar",
    "Y": "polar",
    "N": "polar",
    "Q": "polar",
    "C": "polar",
    # charged
    "K": "positive",
    "R": "positive",
    "H": "positive",
    "D": "negative",
    "E": "negative",
}

HYDROPATHY_CLASS: Dict[str, str] = {
    "A": "hydrophobic",
    "V": "hydrophobic",
    "L": "hydrophobic",
    "I": "hydrophobic",
    "M": "hydrophobic",
    "F": "hydrophobic",
    "W": "hydrophobic",
    "P": "hydrophobic",
    "G": "hydrophobic",
    "C": "hydrophobic",
    "Y": "hydrophobic",
    "S": "hydrophilic",
    "T": "hydrophilic",
    "N": "hydrophilic",
    "Q": "hydrophilic",
    "K": "hydrophilic",
    "R": "hydrophilic",
    "H": "hydrophilic",
    "D": "hydrophilic",
    "E": "hydrophilic",
}

HYDROPATHY_KD: Dict[str, float] = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2
}

MOLECULAR_WEIGHT: Dict[str, float] = {
    "A": 89.1, "R": 174.2, "N": 132.1, "D": 133.1, "C": 121.2,
    "Q": 146.1, "E": 147.1, "G": 75.1, "H": 155.2, "I": 131.2,
    "L": 131.2, "K": 146.2, "M": 149.2, "F": 165.2, "P": 115.1,
    "S": 105.1, "T": 119.1, "W": 204.2, "Y": 181.2, "V": 117.1
}

ISOELECTRIC_POINT: Dict[str, float] = {
    "A": 6.00, "R": 10.76, "N": 5.41, "D": 2.77, "C": 5.07,
    "Q": 5.65, "E": 3.22, "G": 5.97, "H": 7.59, "I": 6.02,
    "L": 5.98, "K": 9.74, "M": 5.74, "F": 5.48, "P": 6.30,
    "S": 5.68, "T": 5.60, "W": 5.89, "Y": 5.66, "V": 5.96
}

START_CODONS = {"ATG", "GTG", "TTG"}  # canonical + common bacterial alternatives
STOP_CODONS = {codon for codon, aa in STANDARD_GENETIC_CODE.items() if aa == "Stop"}


def classify_codon(codon: str) -> Tuple[str, str, str, str, str, str, str, str]:
    """Return (aa, polarity, hydropathy, is_stop, is_start, kd_hydropathy, mw_volume, pi) for a codon token."""
    info = STANDARD_GENETIC_CODE.get(codon)
    if info is None:
        return "", "", "", "", "", "", "", ""

    if info == "Stop":
        return "Stop", "", "", "1", "0", "", "", ""

    polarity = POLARITY_CLASS.get(info, "")
    hydropathy = HYDROPATHY_CLASS.get(info, "")
    is_stop = "1" if codon in STOP_CODONS else "0"
    is_start = "1" if codon in START_CODONS else "0"
    kd = str(HYDROPATHY_KD.get(info, ""))
    mw = str(MOLECULAR_WEIGHT.get(info, ""))
    pi = str(ISOELECTRIC_POINT.get(info, ""))
    return info, polarity, hydropathy, is_stop, is_start, kd, mw, pi


def load_tokens(run_dir: Path) -> list[str]:
    path = run_dir / "itos.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing vocabulary file: {path}")
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate probe labels for a run.")
    parser.add_argument("run_id", nargs="?", help="Run identifier under runs/")
    parser.add_argument(
        "--run_dir", help="Alternative to run_id; path to runs/<RUN_ID>"
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else (RUNS_DIR / str(args.run_id))
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    tokens = load_tokens(run_dir)

    out_path = run_dir / "probe_labels.csv"
    with out_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["token", "aa", "polarity", "hydropathy", "is_stop", "is_start", "kd_hydropathy", "mw_volume", "pi"]
        )
        for token in tokens:
            codon = token.upper()
            if len(codon) == 3 and codon.isalpha():
                aa, polarity, hydropathy, is_stop, is_start, kd, mw, pi = classify_codon(codon)
            else:
                aa = polarity = hydropathy = is_stop = is_start = kd = mw = pi = ""
            writer.writerow([token, aa, polarity, hydropathy, is_stop, is_start, kd, mw, pi])

    print(f"[generate-probe-labels] wrote {out_path}")


if __name__ == "__main__":
    main()
