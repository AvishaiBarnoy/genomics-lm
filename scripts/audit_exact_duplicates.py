#!/usr/bin/env python3
"""Cross-split exact sequence duplicate auditor.

Streams through train and validation (and optionally test) splits before
windowing/tokenization, normalizes DNA sequences to uppercase, and computes
cryptographic hashes. Reports hashes present across multiple splits while
retaining their record IDs without storing sequence copies.

Usage examples:
  # From separate sequence files (one sequence per line, or FASTA):
  python -m scripts.audit_exact_duplicates --train train.txt --val val.txt

  # From paired metadata and sequence files (e.g. pretraining dataset):
  python -m scripts.audit_exact_duplicates \\
      --meta data/processed/corrected/corrected-codonlm-v1/genus/cds_meta.tsv \\
      --dna data/processed/corrected/corrected-codonlm-v1/genus/cds_dna.txt \\
      --report-json audit_report.json

  # Or using --dataset-dir containing cds_meta.tsv and cds_dna.txt:
  python -m scripts.audit_exact_duplicates \\
      --dataset-dir data/processed/corrected/corrected-codonlm-v1/genus \\
      --report-json audit_report.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Set, Tuple


def normalize_dna(sequence: str) -> str:
    """Normalize DNA sequence to uppercase, stripping whitespace and standardizing RNA uracil."""
    return "".join(sequence.split()).upper().replace("U", "T")


def hash_sequence(sequence: str) -> str:
    """Compute SHA-256 digest of normalized sequence."""
    normalized = normalize_dna(sequence)
    return hashlib.sha256(normalized.encode("ascii")).hexdigest()


def stream_fasta(path: Path, split_name: str) -> Generator[Tuple[str, str, str], None, None]:
    """Stream (record_id, split_name, sequence) from a FASTA file."""
    current_id: Optional[str] = None
    seq_chunks: List[str] = []
    line_num = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    yield (current_id, split_name, "".join(seq_chunks))
                    seq_chunks = []
                current_id = line[1:].split()[0] or f"{split_name}_{line_num}"
            else:
                seq_chunks.append(line)
            line_num += 1

        if current_id is not None:
            yield (current_id, split_name, "".join(seq_chunks))


def stream_text_file(path: Path, split_name: str) -> Generator[Tuple[str, str, str], None, None]:
    """Stream (record_id, split_name, sequence) from a plain text file (one seq per line) or FASTA."""
    # Peek at first non-empty char to detect FASTA
    with path.open("r", encoding="utf-8") as f:
        first_line = ""
        for line in f:
            if line.strip():
                first_line = line.strip()
                break

    if first_line.startswith(">"):
        yield from stream_fasta(path, split_name)
    else:
        with path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                seq = line.strip()
                if seq:
                    yield (f"{split_name}_{idx}", split_name, seq)


def stream_dataset_meta_dna(
    meta_path: Path, dna_path: Path
) -> Generator[Tuple[str, str, str], None, None]:
    """Stream (record_id, split_name, sequence) from synchronized meta TSV and dna txt."""
    with meta_path.open("r", encoding="utf-8") as fm, dna_path.open("r", encoding="utf-8") as fd:
        reader = csv.DictReader(fm, delimiter="\t")
        if not reader.fieldnames or "split" not in reader.fieldnames:
            raise ValueError(f"Missing required 'split' column in {meta_path}")
        for line_idx, (meta_row, dna_line) in enumerate(zip_longest(reader, fd)):
            if meta_row is None or dna_line is None:
                raise ValueError(
                    f"Metadata and DNA record counts differ at row {line_idx}"
                )
            split = (meta_row.get("split") or "").strip()
            if not split:
                raise ValueError(f"Missing split at metadata row {line_idx}")
            source_id = (
                meta_row.get("source_id")
                or meta_row.get("record_id")
                or meta_row.get("line_idx")
                or str(line_idx)
            )
            seq = dna_line.strip()
            if seq:
                yield (str(source_id), str(split), seq)


def audit_exact_duplicates(
    records: Iterable[Tuple[str, str, str]],
) -> Dict[str, Any]:
    """Stream records and find exact duplicate hashes across splits.

    Args:
        records: Iterable of (record_id, split_name, raw_sequence).

    Returns:
        Audit report dict containing per-split stats and cross-split duplicate families.
    """
    # Map hash -> split -> set of record_ids
    # Storing only hashes and IDs to minimize memory footprint
    hash_to_split_ids: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    total_records = 0
    records_per_split: Dict[str, int] = defaultdict(int)

    for record_id, split, seq in records:
        h = hash_sequence(seq)
        hash_to_split_ids[h][split].append(record_id)
        records_per_split[split] += 1
        total_records += 1

    # Detect cross-split duplicates
    cross_split_violations: List[Dict[str, Any]] = []
    within_split_duplicate_count = 0

    for digest, split_map in hash_to_split_ids.items():
        splits_present = sorted(split_map.keys())
        if len(splits_present) > 1:
            total_instances = sum(len(ids) for ids in split_map.values())
            cross_split_violations.append(
                {
                    "sha256": digest,
                    "splits": splits_present,
                    "total_instances": total_instances,
                    "records_by_split": {s: sorted(ids) for s, ids in split_map.items()},
                }
            )
        # Check within-split duplicates for awareness
        for s, ids in split_map.items():
            if len(ids) > 1:
                within_split_duplicate_count += len(ids) - 1

    return {
        "status": "passed" if len(cross_split_violations) == 0 else "failed",
        "total_records_audited": total_records,
        "unique_sequences": len(hash_to_split_ids),
        "records_per_split": dict(records_per_split),
        "cross_split_duplicate_count": len(cross_split_violations),
        "within_split_duplicate_instances": within_split_duplicate_count,
        "cross_split_violations": cross_split_violations,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit training and validation splits for exact sequence duplicates."
    )
    group_files = parser.add_argument_group("Input via split files")
    group_files.add_argument("--train", type=Path, help="Path to training sequence file (text or FASTA)")
    group_files.add_argument("--val", type=Path, help="Path to validation sequence file (text or FASTA)")
    group_files.add_argument("--test", type=Path, help="Optional path to test sequence file")

    group_dataset = parser.add_argument_group("Input via dataset directory or meta/dna")
    group_dataset.add_argument("--dataset-dir", type=Path, help="Path to directory containing cds_meta.tsv and cds_dna.txt")
    group_dataset.add_argument("--meta", type=Path, help="Path to cds_meta.tsv with 'split' column")
    group_dataset.add_argument("--dna", type=Path, help="Path to cds_dna.txt")

    parser.add_argument("--report-json", type=Path, help="Optional path to write audit report JSON")
    parser.add_argument(
        "--fail-on-duplicate",
        action="store_true",
        help="Exit with non-zero status if cross-split duplicates are detected",
    )

    args = parser.parse_args(argv)

    # Determine input mode
    streams: List[Iterable[Tuple[str, str, str]]] = []

    if args.dataset_dir:
        meta_p = args.dataset_dir / "cds_meta.tsv"
        dna_p = args.dataset_dir / "cds_dna.txt"
        if not meta_p.exists() or not dna_p.exists():
            parser.error(f"Directory {args.dataset_dir} must contain cds_meta.tsv and cds_dna.txt")
        streams.append(stream_dataset_meta_dna(meta_p, dna_p))

    elif args.meta or args.dna:
        if not (args.meta and args.dna):
            parser.error("Both --meta and --dna must be provided together.")
        streams.append(stream_dataset_meta_dna(args.meta, args.dna))

    elif args.train or args.val:
        if not (args.train and args.val):
            parser.error("Both --train and --val are required when specifying split files.")
        streams.append(stream_text_file(args.train, "train"))
        streams.append(stream_text_file(args.val, "val"))
        if args.test:
            streams.append(stream_text_file(args.test, "test"))
    else:
        parser.error(
            "Please provide either (--train and --val) or (--dataset-dir) or (--meta and --dna)."
        )

    def record_generator():
        for st in streams:
            yield from st

    print("[exact-duplicate-audit] Streaming and hashing sequences...")
    report = audit_exact_duplicates(record_generator())

    print(f"[exact-duplicate-audit] Total records audited: {report['total_records_audited']:,}")
    print(f"[exact-duplicate-audit] Unique sequences:       {report['unique_sequences']:,}")
    print(f"[exact-duplicate-audit] Records per split:      {report['records_per_split']}")
    print(f"[exact-duplicate-audit] Cross-split duplicates: {report['cross_split_duplicate_count']:,}")

    if report["cross_split_duplicate_count"] > 0:
        print("[!] Cross-split duplicate families found:")
        for v in report["cross_split_violations"][:10]:
            print(f"  - sha256: {v['sha256'][:12]}... in splits {v['splits']}")
            for s, ids in v["records_by_split"].items():
                print(f"      {s} ({len(ids)} ids): {ids[:5]}")
        if len(report["cross_split_violations"]) > 10:
            print(f"  ... and {len(report['cross_split_violations']) - 10} more.")

    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2) + "\n")
        print(f"[exact-duplicate-audit] Wrote report to {args.report_json}")

    if args.fail_on_duplicate and report["cross_split_duplicate_count"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
