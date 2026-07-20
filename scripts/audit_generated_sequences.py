#!/usr/bin/env python3
"""Audit generated CDS novelty against the training source-record corpus."""

from __future__ import annotations

import argparse
from pathlib import Path

from Bio import SeqIO

from src.codonlm.leakage_audit import audit_generated_sequences


def _read_fasta(path: Path) -> list[dict[str, str]]:
    return [
        {"source_id": str(record.id), "sequence": str(record.seq)}
        for record in SeqIO.parse(path, "fasta")
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-fasta", type=Path, required=True)
    parser.add_argument("--generated-fasta", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nucleotide-window", type=int, default=30)
    parser.add_argument("--protein-window", type=int, default=10)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--mmseqs-executable", default="mmseqs")
    args = parser.parse_args()

    report = audit_generated_sequences(
        _read_fasta(args.train_fasta),
        _read_fasta(args.generated_fasta),
        args.output,
        nucleotide_window=args.nucleotide_window,
        protein_window=args.protein_window,
        threads=args.threads,
        executable=args.mmseqs_executable,
    )
    print(
        f"[generated-audit] wrote {report['generated_record_count']} records to {args.output}"
    )


if __name__ == "__main__":
    main()
