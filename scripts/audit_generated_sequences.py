#!/usr/bin/env python3
"""Audit generated CDS novelty against the training source-record corpus."""

from __future__ import annotations

import argparse
import csv
import json
from itertools import zip_longest
from pathlib import Path

from Bio import SeqIO

from src.codonlm.leakage_audit import audit_generated_sequences
from src.codonlm.dataset_manifest import manifest_artifact_path
from src.codonlm.evaluation_provenance import (
    artifact_provenance,
    bind_dataset_manifest,
)


def _read_fasta(path: Path) -> list[dict[str, str]]:
    return [
        {"source_id": str(record.id), "sequence": str(record.seq)}
        for record in SeqIO.parse(path, "fasta")
    ]


def _read_manifest_training(manifest_path: Path) -> tuple[list[dict[str, str]], dict]:
    manifest, provenance = bind_dataset_manifest(manifest_path)
    resolved_manifest = manifest_path.expanduser().resolve()
    metadata_path = manifest_artifact_path(
        manifest, resolved_manifest, "source_metadata"
    )
    dna_path = manifest_artifact_path(manifest, resolved_manifest, "source_dna")
    records = []
    with metadata_path.open(newline="") as metadata_handle, dna_path.open() as dna_handle:
        metadata_rows = csv.DictReader(metadata_handle, delimiter="\t")
        for index, pair in enumerate(zip_longest(metadata_rows, dna_handle)):
            row, sequence = pair
            if row is None or sequence is None:
                raise ValueError("source metadata and DNA artifacts have different row counts")
            if int(row["line_idx"]) != index:
                raise ValueError(
                    f"source metadata line_idx mismatch at row {index}: {row['line_idx']}"
                )
            if row["split"] == "train":
                records.append(
                    {"source_id": row["source_id"], "sequence": sequence.strip()}
                )
    if not records:
        raise ValueError("frozen manifest contains no training source records")
    provenance["training_source"] = {
        "selection": "source_metadata.split == train",
        "record_count": len(records),
        "source_metadata": artifact_provenance(metadata_path),
        "source_dna": artifact_provenance(dna_path),
    }
    return records, provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    training = parser.add_mutually_exclusive_group(required=True)
    training.add_argument("--train-fasta", type=Path)
    training.add_argument(
        "--manifest",
        type=Path,
        help="Frozen manifest from which train-only source records are derived.",
    )
    parser.add_argument("--generated-fasta", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nucleotide-window", type=int, default=30)
    parser.add_argument("--protein-window", type=int, default=10)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--mmseqs-executable", default="mmseqs")
    parser.add_argument("--minimap2-executable", default="minimap2")
    parser.add_argument("--nucleotide-preset", default="asm20")
    parser.add_argument(
        "--split-memory-limit",
        default=None,
        help="MMseqs2 memory per target-database split, for example 3G.",
    )
    parser.add_argument(
        "--training-batch-size",
        type=int,
        default=5000,
        help="Maximum training records in each explicit MMseqs2 target batch.",
    )
    args = parser.parse_args()

    if args.manifest is not None:
        training_records, manifest_provenance = _read_manifest_training(args.manifest)
    else:
        training_records = _read_fasta(args.train_fasta)
        manifest_provenance = {"status": "legacy_unverified"}
    report = audit_generated_sequences(
        training_records,
        _read_fasta(args.generated_fasta),
        args.output,
        nucleotide_window=args.nucleotide_window,
        protein_window=args.protein_window,
        threads=args.threads,
        executable=args.mmseqs_executable,
        nucleotide_executable=args.minimap2_executable,
        nucleotide_preset=args.nucleotide_preset,
        split_memory_limit=args.split_memory_limit,
        training_batch_size=args.training_batch_size,
    )
    report["dataset_manifest"] = manifest_provenance
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        f"[generated-audit] wrote {report['generated_record_count']} records to {args.output}"
    )


if __name__ == "__main__":
    main()
