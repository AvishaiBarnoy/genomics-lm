#!/usr/bin/env python3
"""Audit a downstream test set against frozen CodonLM training records."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.codonlm.leakage_audit import audit_source_records


def _load_pretraining(meta_path: Path, dna_path: Path) -> list[dict]:
    sequences = dna_path.read_text().splitlines()
    records = []
    with meta_path.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row.get("split") != "train":
                continue
            line_idx = int(row["line_idx"])
            if line_idx >= len(sequences):
                raise ValueError(f"line_idx {line_idx} exceeds {dna_path}")
            records.append(
                {
                    "source_id": f"pretraining:{row['source_id']}",
                    "split": "train",
                    "sequence": sequences[line_idx],
                }
            )
    return records


def _load_downstream(path: Path, id_column: str, sequence_column: str) -> list[dict]:
    records = []
    with path.open(newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            sequence = row.get(sequence_column, "").strip()
            if not sequence:
                continue
            source_id = row.get(id_column, "").strip() or f"row-{index}"
            records.append(
                {
                    "source_id": f"downstream:{source_id}",
                    "split": "test",
                    "sequence": sequence,
                }
            )
    return records


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cds-meta", type=Path, required=True)
    parser.add_argument("--cds-dna", type=Path, required=True)
    parser.add_argument("--downstream-seqs", type=Path, required=True)
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--sequence-column", default="seq")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mmseqs-executable", default="mmseqs")
    parser.add_argument("--minimap2-executable", default="minimap2")
    parser.add_argument("--min-protein-identity", type=float, default=0.3)
    parser.add_argument("--min-coverage", type=float, default=0.8)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--nearest-query-batch-size", type=int, default=4096)
    args = parser.parse_args(argv)

    pretraining = _load_pretraining(args.cds_meta, args.cds_dna)
    downstream = _load_downstream(
        args.downstream_seqs, args.id_column, args.sequence_column
    )
    if not pretraining:
        raise ValueError("no frozen pretraining-train records were loaded")
    if not downstream:
        raise ValueError("no downstream test records were loaded")
    report = audit_source_records(
        pretraining + downstream,
        args.output,
        min_protein_identity=args.min_protein_identity,
        min_coverage=args.min_coverage,
        threads=args.threads,
        executable=args.mmseqs_executable,
        protein_homology_policy="report",
        nucleotide_executable=args.minimap2_executable,
        nearest_query_batch_size=args.nearest_query_batch_size,
    )
    report["scope"] = {
        "pretraining_train_records": len(pretraining),
        "downstream_test_records": len(downstream),
        "downstream_sequences": str(args.downstream_seqs.resolve()),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"[downstream-audit] status={report['status']} "
        f"exact={report['exact_duplicates']['count']} "
        f"protein_clusters={report['protein_homology']['cross_split_cluster_count']}"
    )


if __name__ == "__main__":
    main()
