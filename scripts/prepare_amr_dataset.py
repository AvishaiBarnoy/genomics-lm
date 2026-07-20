"""
scripts/prepare_amr_dataset.py — CARD AMR Dataset Preparation for CodonLM Probe

Reads:
  data/raw/amr_card/nucleotide_fasta_protein_homolog_model.fasta
  data/raw/amr_card/aro_index.tsv

Outputs under <out_dir>/<protocol>/:
  train_amr.csv, test_amr.csv, train_amr_seqs.csv, test_amr_seqs.csv
  amr_label_map.json, split_assignments.tsv, split_report.json

Usage:
  python -m scripts.prepare_amr_dataset --out_dir outputs/amr \
    --protocol annotation_family_held_out
  python -m scripts.prepare_amr_dataset --out_dir outputs/amr \
    --protocol protein_cluster_held_out
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np
from src.codonlm.leakage_audit import translate_cds

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FASTA_PATH = Path("data/raw/amr_card/nucleotide_fasta_protein_homolog_model.fasta")
ARO_INDEX_PATH = Path("data/raw/amr_card/aro_index.tsv")
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


def _load_aro_metadata(path: Path) -> dict[str, tuple[str, str]]:
    """Returns {ARO_accession -> (normalized_drug_class, amr_gene_family)}."""
    mapping: dict[str, tuple[str, str]] = {}
    with path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            aro = row.get("ARO Accession", "").strip()
            drug_raw = row.get("Drug Class", "").strip()
            family = row.get("AMR Gene Family", "").strip()
            if not aro or not drug_raw:
                continue
            normalized = _normalize_drug_class(drug_raw)
            if normalized:
                if not family:
                    family = aro
                mapping[aro] = (normalized, family)
    return mapping


def _parse_fasta(path: Path) -> Iterator[tuple[str, str, str]]:
    """Yield (header, sequence) from a nucleotide FASTA."""
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


def _stratified_group_split(
    records: list[dict],
    group_key: str = "family",
    test_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Approximate a stratified split without dividing the selected groups."""
    rng = np.random.default_rng(seed)
    
    # 1. Map each unique family to its primary drug_class
    family_to_class_counts = defaultdict(lambda: Counter())
    for r in records:
        family_to_class_counts[r[group_key]][r["drug_class"]] += 1
        
    family_primary_class = {}
    for family, class_counts in family_to_class_counts.items():
        family_primary_class[family] = class_counts.most_common(1)[0][0]
        
    # 2. Group unique families by their primary drug class
    families_by_class = defaultdict(list)
    for family, cls in family_primary_class.items():
        families_by_class[cls].append(family)
        
    # 3. For each drug class, split the families
    test_families = set()
    for cls, families in families_by_class.items():
        families = list(families)
        rng.shuffle(families)
        total_records_in_class = sum(1 for r in records if r["drug_class"] == cls)
        target_test_records = int(total_records_in_class * test_fraction)
        
        current_test_records = 0
        for family in families:
            family_record_count = sum(1 for r in records if r[group_key] == family and r["drug_class"] == cls)
            if current_test_records < target_test_records or current_test_records == 0:
                test_families.add(family)
                current_test_records += family_record_count
            else:
                break
                
    # 4. Assign records based on family split
    train_records = []
    test_records = []
    for r in records:
        if r[group_key] in test_families:
            test_records.append(r)
        else:
            train_records.append(r)
            
    # Check overlap sanity
    train_f = {r[group_key] for r in train_records}
    test_f = {r[group_key] for r in test_records}
    assert train_f.isdisjoint(test_f), f"[split] overlap detected for {group_key}"
    
    return train_records, test_records


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cluster_proteins(records, work_dir, executable, min_identity, coverage, threads):
    resolved = shutil.which(executable)
    if resolved is None:
        raise RuntimeError(f"MMseqs2 executable not found: {executable}")
    work_dir.mkdir(parents=True, exist_ok=True)
    protein_fasta = work_dir / "amr_proteins.fasta"
    with protein_fasta.open("w") as handle:
        for index, record in enumerate(records):
            source_key = f"record-{index}"
            record["source_key"] = source_key
            protein = translate_cds(record["dna"])
            if not protein or "X" in protein:
                raise ValueError(f"invalid translated protein for {record['id']}")
            handle.write(f">{source_key}\n{protein}\n")
    prefix, tmp = work_dir / "clusters", work_dir / "tmp"
    command = [
        resolved, "easy-cluster", str(protein_fasta), str(prefix), str(tmp),
        "--min-seq-id", str(min_identity), "-c", str(coverage),
        "--cov-mode", "0", "--cluster-mode", "0", "--threads", str(threads),
    ]
    version_result = subprocess.run(
        [resolved, "version"], check=True, capture_output=True, text=True
    )
    subprocess.run(command, check=True, capture_output=True, text=True)
    member_to_cluster = {}
    cluster_path = Path(f"{prefix}_cluster.tsv")
    with cluster_path.open() as handle:
        for line in handle:
            representative, member = line.rstrip("\n").split("\t")[:2]
            member_to_cluster[member] = representative
    missing = [record["source_key"] for record in records if record["source_key"] not in member_to_cluster]
    if missing:
        raise RuntimeError(f"MMseqs2 output omitted records: {missing[:5]}")
    for record in records:
        record["protein_cluster"] = member_to_cluster[record["source_key"]]
    return {
        "tool": {"executable": resolved, "version": (version_result.stdout or version_result.stderr).strip()},
        "command": command,
        "thresholds": {"min_sequence_identity": min_identity, "min_coverage": coverage, "coverage_mode": 0},
        "protein_fasta": {"path": str(protein_fasta), "sha256": _sha256(protein_fasta)},
        "cluster_assignments": {
            "path": str(cluster_path),
            "sha256": _sha256(cluster_path),
        },
        "cluster_count": len(set(member_to_cluster.values())),
    }


def _distribution_report(train, test, protocol, requested_fraction):
    classes = sorted({record["drug_class"] for record in train + test})
    def summarize(records):
        return {
            "records": len(records),
            "records_per_class": dict(sorted(Counter(r["drug_class"] for r in records).items())),
            "families_per_class": {
                cls: len({r["family"] for r in records if r["drug_class"] == cls}) for cls in classes
            },
            "clusters_per_class": {
                cls: len({r.get("protein_cluster") for r in records if r["drug_class"] == cls and r.get("protein_cluster")}) for cls in classes
            },
        }
    train_summary, test_summary = summarize(train), summarize(test)
    train_classes = {r["drug_class"] for r in train}
    test_classes = {r["drug_class"] for r in test}
    return {
        "protocol": protocol,
        "requested_test_fraction": requested_fraction,
        "achieved_test_fraction": len(test) / max(1, len(train) + len(test)),
        "train": train_summary,
        "test": test_summary,
        "missing_train_classes": sorted(set(classes) - train_classes),
        "missing_test_classes": sorted(set(classes) - test_classes),
        "mixed_class_clusters": sum(
            len({r["drug_class"] for r in train + test if r.get("protein_cluster") == cluster}) > 1
            for cluster in {r.get("protein_cluster") for r in train + test if r.get("protein_cluster")}
        ),
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", default=str(FASTA_PATH))
    ap.add_argument("--aro_index", default=str(ARO_INDEX_PATH))
    ap.add_argument("--out_dir", required=True)
    ap.add_argument(
        "--protocol",
        choices=("annotation_family_held_out", "protein_cluster_held_out"),
        default="annotation_family_held_out",
    )
    ap.add_argument("--mmseqs-executable", default="mmseqs")
    ap.add_argument("--min-protein-identity", type=float, default=0.3)
    ap.add_argument("--min-coverage", type=float, default=0.8)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--min_examples", type=int, default=60,
                    help="Minimum gene examples per drug class to include")
    ap.add_argument("--top_n_classes", type=int, default=8,
                    help="Keep at most N most-common drug classes")
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    if not 0.0 < args.test_frac < 1.0:
        ap.error("--test_frac must be between 0 and 1")
    if not 0.0 <= args.min_protein_identity <= 1.0:
        ap.error("--min-protein-identity must be between 0 and 1")
    if not 0.0 < args.min_coverage <= 1.0:
        ap.error("--min-coverage must be in (0, 1]")

    fasta_path = Path(args.fasta)
    aro_path = Path(args.aro_index)
    out_root = Path(args.out_dir)
    out_dir = out_root / args.protocol
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load ARO → drug class & family mapping
    print("[amr] Loading ARO index...")
    aro_map = _load_aro_metadata(aro_path)
    print(f"[amr] ARO entries with known drug class: {len(aro_map)}")

    # 2. Parse FASTA and join to drug class & family
    print(f"[amr] Parsing FASTA: {fasta_path}")
    records: list[dict] = []
    stats = Counter(skipped_no_aro=0, skipped_no_class=0, skipped_too_short=0, skipped_invalid=0, kept=0)

    for header, seq_parts in _parse_fasta(fasta_path):
        seq = "".join(seq_parts)
        aro = _extract_aro(header)
        if not aro:
            stats["skipped_no_aro"] += 1
            continue
        metadata = aro_map.get(aro)
        if not metadata:
            stats["skipped_no_class"] += 1
            continue
        drug_class, family = metadata
        codons = _to_codons(seq)
        if codons is None:
            stats["skipped_too_short"] += 1
            continue
        if len(codons) > MAX_CODONS:
            codons = codons[:MAX_CODONS]
        records.append({
            "id": header.split("|")[1] if "|" in header else header[:40],
            "aro": aro,
            "family": family,
            "sequence": " ".join(codons),
            "dna": "".join(codons),
            "n_codons": len(codons),
            "drug_class": drug_class,
        })
        stats["kept"] += 1

    print(f"[amr] Stats: {dict(stats)}")

    # 3. Filter to top-N classes with >= min_examples
    class_counts = Counter(r["drug_class"] for r in records)
    print("\n[amr] Raw drug class distribution:")
    for cls, cnt in class_counts.most_common(20):
        print(f"  {cls:30s}: {cnt}")

    eligible = {cls for cls, cnt in class_counts.items() if cnt >= args.min_examples}
    top_classes = [cls for cls, _ in class_counts.most_common(args.top_n_classes) if cls in eligible]

    print(f"\n[amr] Keeping {len(top_classes)} classes (min_examples={args.min_examples}):")
    if not top_classes:
        raise ValueError("no drug classes satisfy the requested filters")
    for i, cls in enumerate(top_classes):
        print(f"  [{i}] {cls}: {class_counts[cls]} examples")

    records = [r for r in records if r["drug_class"] in top_classes]
    label_map = {cls: i for i, cls in enumerate(top_classes)}
    for r in records:
        r["label"] = label_map[r["drug_class"]]

    clustering = None
    group_key = "family"
    if args.protocol == "protein_cluster_held_out":
        clustering = _cluster_proteins(
            records,
            out_dir / "clustering",
            args.mmseqs_executable,
            args.min_protein_identity,
            args.min_coverage,
            args.threads,
        )
        group_key = "protein_cluster"

    # 4. Split whole annotation families or computed protein clusters.
    train_records, test_records = _stratified_group_split(
        records, group_key, args.test_frac, args.seed
    )
    print(f"\n[amr] Train: {len(train_records)}  |  Test: {len(test_records)}")

    # 5. Write CSVs
    fieldnames = [
        "id", "aro", "family", "protein_cluster", "sequence", "n_codons",
        "drug_class", "label",
    ]
    for split_name, split_records in [("train_amr", train_records), ("test_amr", test_records)]:
        out_path = out_dir / f"{split_name}.csv"
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(
                {name: record.get(name, "") for name in fieldnames}
                for record in split_records
            )
        print(f"[amr] Wrote {out_path} ({len(split_records)} rows)")

    # K-mer inputs live with the selected protocol and never mutate research data implicitly.
    proc_dir = out_dir
    for split_name, split_records in [("train_amr", train_records), ("test_amr", test_records)]:
        seqs_path = proc_dir / f"{split_name}_seqs.csv"
        with seqs_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "seq"])
            for r in split_records:
                raw_seq = r["sequence"].replace(" ", "")
                writer.writerow([r["id"], raw_seq])
        print(f"[amr] Wrote {seqs_path} ({len(split_records)} rows)")

    # Label map
    label_map_path = out_dir / "amr_label_map.json"
    with label_map_path.open("w") as f:
        json.dump({"label_to_id": label_map, "id_to_label": {str(v): k for k, v in label_map.items()}}, f, indent=2)
    print(f"[amr] Label map → {label_map_path}")

    # 6. Summary
    print("\n[amr] ✅ Dataset ready.")
    print(f"  Classes: {len(top_classes)}")
    print(f"  Train:   {len(train_records)}")
    print(f"  Test:    {len(test_records)}")
    random_baseline = 1.0 / len(top_classes)
    print(f"  Random baseline accuracy: {random_baseline:.1%} (1/{len(top_classes)})")

    report = _distribution_report(
        train_records, test_records, args.protocol, args.test_frac
    )
    report.update(
        {
            "seed": args.seed,
            "group_key": group_key,
            "inputs": {
                "fasta": {"path": str(fasta_path.resolve()), "sha256": _sha256(fasta_path)},
                "aro_index": {"path": str(aro_path.resolve()), "sha256": _sha256(aro_path)},
            },
            "clustering": clustering,
            "filtering": dict(stats),
            "class_normalization": CLASS_NORMALIZATION,
        }
    )
    (out_dir / "split_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    with (out_dir / "split_assignments.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["id", "drug_class", "family", "protein_cluster", "split"])
        for split, split_records in (("train", train_records), ("test", test_records)):
            for record in split_records:
                writer.writerow([
                    record["id"], record["drug_class"], record["family"],
                    record.get("protein_cluster", ""), split,
                ])


if __name__ == "__main__":
    main()
