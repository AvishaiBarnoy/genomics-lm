"""
scripts/prepare_amr_dataset.py — CARD AMR Dataset Preparation for CodonLM Probe

Reads:
  data/raw/amr_card/nucleotide_fasta_protein_homolog_model.fasta
  data/raw/amr_card/aro_index.tsv

Outputs:
  data/labels/train_amr.csv   (columns: gene_id, sequence, drug_class, label_id)
  data/labels/test_amr.csv
  data/labels/amr_label_map.json

Usage:
  python -m scripts.prepare_amr_dataset
  python -m scripts.prepare_amr_dataset --min_examples 80 --top_n_classes 6
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FASTA_PATH = Path("data/raw/amr_card/nucleotide_fasta_protein_homolog_model.fasta")
ARO_INDEX_PATH = Path("data/raw/amr_card/aro_index.tsv")
OUT_DIR = Path("data/labels")

# Only consider sequences that fit in the model's codon context window
MIN_CODONS = 20    # too short → not meaningful
MAX_CODONS = 500   # fits within 512-codon context

# Codon alphabet (must match tokenizer)
VALID_BASES = set("ATCG")

# Broad drug class normalization — collapse synonymous / multi-label names
CLASS_NORMALIZATION = {
    "beta-lactam antibiotic": "beta-lactam",
    "cephalosporin": "beta-lactam",
    "carbapenem": "beta-lactam",
    "penam": "beta-lactam",
    "penem": "beta-lactam",
    "monobactam": "beta-lactam",
    "aminoglycoside antibiotic": "aminoglycoside",
    "tetracycline antibiotic": "tetracycline",
    "fluoroquinolone antibiotic": "fluoroquinolone",
    "macrolide antibiotic": "macrolide",
    "lincosamide antibiotic": "macrolide/MLS",
    "streptogramin antibiotic": "macrolide/MLS",
    "rifamycin antibiotic": "rifamycin",
    "colistin": "polymyxin",
    "polymyxin antibiotic": "polymyxin",
    "sulfonamide antibiotic": "sulfonamide",
    "trimethoprim antibiotic": "sulfonamide",
    "chloramphenicol antibiotic": "chloramphenicol",
    "glycopeptide antibiotic": "glycopeptide",
}


def _normalize_drug_class(raw: str) -> str | None:
    """Normalize a (possibly multi-drug) drug class field to a single canonical class.

    Returns the first matching class, or None if no known class matches.
    """
    parts = [p.strip().lower() for p in raw.split(";")]
    for part in parts:
        for key, normalized in CLASS_NORMALIZATION.items():
            if key in part:
                return normalized
    return None


def _load_aro_drug_classes(path: Path) -> dict[str, str]:
    """Returns {ARO_accession -> normalized_drug_class}."""
    mapping: dict[str, str] = {}
    with path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            aro = row.get("ARO Accession", "").strip()
            drug_raw = row.get("Drug Class", "").strip()
            if not aro or not drug_raw:
                continue
            normalized = _normalize_drug_class(drug_raw)
            if normalized:
                mapping[aro] = normalized
    return mapping


def _parse_fasta(path: Path) -> Iterator[tuple[str, str, str]]:
    """Yield (header, aro_accession, sequence) from a nucleotide FASTA."""
    header, seq_parts = None, []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header and seq_parts:
                    yield header, seq_parts
                header = line[1:]
                seq_parts = []
            else:
                seq_parts.append(line.upper())
    if header and seq_parts:
        yield header, seq_parts


def _extract_aro(header: str) -> str | None:
    """Extract ARO accession from FASTA header like: ...ARO:3002999|..."""
    m = re.search(r"(ARO:\d+)", header)
    return m.group(1) if m else None


def _to_codons(seq: str) -> list[str] | None:
    """Convert nucleotide sequence to codon list. Returns None if invalid."""
    # Strip to only ATCG
    seq = re.sub(r"[^ATCG]", "", seq)
    if len(seq) % 3 != 0:
        seq = seq[:len(seq) - (len(seq) % 3)]  # truncate to codon boundary
    if len(seq) < MIN_CODONS * 3:
        return None
    codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
    # Validate all codons are pure ATCG
    if any(len(c) != 3 or not all(b in VALID_BASES for b in c) for c in codons):
        return None
    return codons


def _stratified_split(
    records: list[dict],
    test_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Stratified train/test split by drug_class label."""
    rng = np.random.default_rng(seed)
    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_class[r["drug_class"]].append(r)

    train, test = [], []
    for cls, items in by_class.items():
        items = list(items)
        rng.shuffle(items)
        n_test = max(1, int(len(items) * test_fraction))
        test.extend(items[:n_test])
        train.extend(items[n_test:])
    return train, test


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", default=str(FASTA_PATH))
    ap.add_argument("--aro_index", default=str(ARO_INDEX_PATH))
    ap.add_argument("--out_dir", default=str(OUT_DIR))
    ap.add_argument("--min_examples", type=int, default=60,
                    help="Minimum gene examples per drug class to include")
    ap.add_argument("--top_n_classes", type=int, default=8,
                    help="Keep at most N most-common drug classes")
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    fasta_path = Path(args.fasta)
    aro_path = Path(args.aro_index)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Load ARO → drug class mapping
    # -----------------------------------------------------------------------
    print("[amr] Loading ARO index...")
    aro_map = _load_aro_drug_classes(aro_path)
    print(f"[amr] ARO entries with known drug class: {len(aro_map)}")

    # -----------------------------------------------------------------------
    # 2. Parse FASTA and join to drug class
    # -----------------------------------------------------------------------
    print(f"[amr] Parsing FASTA: {fasta_path}")
    records: list[dict] = []
    stats = Counter(skipped_no_aro=0, skipped_no_class=0, skipped_too_short=0, skipped_invalid=0, kept=0)

    for header, seq_parts in _parse_fasta(fasta_path):
        seq = "".join(seq_parts)
        aro = _extract_aro(header)
        if not aro:
            stats["skipped_no_aro"] += 1
            continue
        drug_class = aro_map.get(aro)
        if not drug_class:
            stats["skipped_no_class"] += 1
            continue
        codons = _to_codons(seq)
        if codons is None:
            stats["skipped_too_short"] += 1
            continue
        # Truncate to MAX_CODONS
        if len(codons) > MAX_CODONS:
            codons = codons[:MAX_CODONS]
        records.append({
            "id": header.split("|")[1] if "|" in header else header[:40],
            "aro": aro,
            "sequence": " ".join(codons),  # space-separated codon tokens
            "n_codons": len(codons),
            "drug_class": drug_class,
        })
        stats["kept"] += 1

    print(f"[amr] Stats: {dict(stats)}")

    # -----------------------------------------------------------------------
    # 3. Filter to top-N classes with >= min_examples
    # -----------------------------------------------------------------------
    class_counts = Counter(r["drug_class"] for r in records)
    print("\n[amr] Raw drug class distribution:")
    for cls, cnt in class_counts.most_common(20):
        print(f"  {cls:30s}: {cnt}")

    eligible = {cls for cls, cnt in class_counts.items() if cnt >= args.min_examples}
    top_classes = [cls for cls, _ in class_counts.most_common(args.top_n_classes) if cls in eligible]

    print(f"\n[amr] Keeping {len(top_classes)} classes (min_examples={args.min_examples}):")
    for i, cls in enumerate(top_classes):
        print(f"  [{i}] {cls}: {class_counts[cls]} examples")

    records = [r for r in records if r["drug_class"] in top_classes]
    label_map = {cls: i for i, cls in enumerate(top_classes)}
    for r in records:
        r["label"] = label_map[r["drug_class"]]

    # -----------------------------------------------------------------------
    # 4. Stratified split
    # -----------------------------------------------------------------------
    train_records, test_records = _stratified_split(records, args.test_frac, args.seed)
    print(f"\n[amr] Train: {len(train_records)}  |  Test: {len(test_records)}")

    # -----------------------------------------------------------------------
    # 5. Write CSVs
    # -----------------------------------------------------------------------
    fieldnames = ["id", "aro", "sequence", "n_codons", "drug_class", "label"]
    for split_name, split_records in [("train_amr", train_records), ("test_amr", test_records)]:
        out_path = out_dir / f"{split_name}.csv"
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(split_records)
        print(f"[amr] Wrote {out_path} ({len(split_records)} rows)")

    # Label map
    label_map_path = out_dir / "amr_label_map.json"
    with label_map_path.open("w") as f:
        json.dump({"label_to_id": label_map, "id_to_label": {str(v): k for k, v in label_map.items()}}, f, indent=2)
    print(f"[amr] Label map → {label_map_path}")

    # -----------------------------------------------------------------------
    # 6. Summary
    # -----------------------------------------------------------------------
    print("\n[amr] ✅ Dataset ready.")
    print(f"  Classes: {len(top_classes)}")
    print(f"  Train:   {len(train_records)}")
    print(f"  Test:    {len(test_records)}")
    random_baseline = 1.0 / len(top_classes)
    print(f"  Random baseline accuracy: {random_baseline:.1%} (1/{len(top_classes)})")


if __name__ == "__main__":
    main()
