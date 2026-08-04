#!/usr/bin/env python3
"""Build a provenance-bound, protein-cluster-held-out ProteinCritic dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from src.protein_lm.corrected_dataset import (
    assign_clusters,
    eligible_labels,
    group_by_sequence,
    normalize_protein,
    split_report,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_annotation_records(proteins_path: Path, metadata_path: Path) -> list[dict]:
    proteins = json.loads(proteins_path.read_text())
    metadata = {}
    with metadata_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            metadata.setdefault(str(row.get("ncbi_id", "")).strip(), row)

    records = []
    for protein_id, payload in proteins.items():
        row = metadata.get(protein_id)
        if row is None:
            continue
        try:
            sequence = normalize_protein(payload.get("sequence", ""))
        except ValueError:
            continue
        pfam_values = [
            value.strip()
            for value in str(row.get("pfam", "")).split(";")
            if value.strip()
        ]
        ec = str(row.get("ec", "")).strip()
        ec_label = (
            int(ec[0]) if ec and ec[0].isdigit() and 1 <= int(ec[0]) <= 7 else None
        )
        pfam_label = pfam_values[0] if pfam_values else None
        if pfam_label is None and ec_label is None:
            continue
        records.append(
            {
                "sequence": sequence,
                "source": "genome_uniprot_annotation",
                "source_ids": [protein_id],
                "pfam_label": pfam_label,
                "ec_label": ec_label,
                "stability_score": None,
            }
        )
    return records


def load_stability_records(path: Path) -> list[dict]:
    records = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                sequence = normalize_protein(row.get("aa_seq", ""))
                score = float(row["deltaG"])
            except (ValueError, KeyError):
                continue
            records.append(
                {
                    "sequence": sequence,
                    "source": "megascale_delta_g",
                    "source_ids": [str(row.get("name", ""))],
                    "pfam_label": None,
                    "ec_label": None,
                    "stability_score": score,
                }
            )
    return records


def cluster_records(records: list[dict], work_dir: Path, args) -> dict:
    executable = shutil.which(args.mmseqs_executable)
    if executable is None:
        raise RuntimeError(f"MMseqs2 executable not found: {args.mmseqs_executable}")
    work_dir.mkdir(parents=True, exist_ok=True)
    fasta = work_dir / "proteins.fasta"
    with fasta.open("w") as handle:
        for index, record in enumerate(records):
            key = f"record-{index}"
            record["cluster_key"] = key
            handle.write(f">{key}\n{record['sequence']}\n")
    prefix = work_dir / "clusters"
    command = [
        executable,
        "easy-cluster",
        str(fasta),
        str(prefix),
        str(work_dir / "tmp"),
        "--min-seq-id",
        str(args.min_sequence_identity),
        "-c",
        str(args.min_coverage),
        "--cov-mode",
        "0",
        "--cluster-mode",
        "0",
        "--threads",
        str(args.threads),
    ]
    version = subprocess.run(
        [executable, "version"], check=True, capture_output=True, text=True
    )
    subprocess.run(command, check=True, capture_output=True, text=True)
    assignments = {}
    cluster_path = Path(f"{prefix}_cluster.tsv")
    with cluster_path.open() as handle:
        for line in handle:
            representative, member = line.rstrip("\n").split("\t")[:2]
            assignments[member] = representative
    missing = [
        record["cluster_key"]
        for record in records
        if record["cluster_key"] not in assignments
    ]
    if missing:
        raise RuntimeError(f"MMseqs2 omitted {len(missing)} records")
    for record in records:
        record["protein_cluster"] = assignments[record["cluster_key"]]
    return {
        "tool": {
            "executable": executable,
            "version": (version.stdout or version.stderr).strip(),
        },
        "command": command,
        "thresholds": {
            "minimum_sequence_identity": args.min_sequence_identity,
            "minimum_coverage": args.min_coverage,
            "coverage_mode": 0,
        },
        "protein_fasta_sha256": sha256(fasta),
        "cluster_assignments_sha256": sha256(cluster_path),
        "cluster_count": len(set(assignments.values())),
    }


def write_jsonl(
    path: Path, records: list[dict], pfam_vocab: dict, ec_vocab: dict
) -> None:
    with path.open("w") as handle:
        for record in records:
            output = {
                "record_id": record["record_id"],
                "sequence": record["sequence"],
                "source": record["source"],
                "source_ids": record["source_ids"],
                "protein_cluster": record["protein_cluster"],
                "pfam_id": pfam_vocab.get(record.get("pfam_label"), -1),
                "ec_id": ec_vocab.get(record.get("ec_label"), -1),
                "stability_score": record.get("stability_score"),
            }
            handle.write(json.dumps(output, sort_keys=True) + "\n")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protein-records", type=Path, required=True)
    parser.add_argument("--annotation-metadata", type=Path, required=True)
    parser.add_argument("--stability-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--mmseqs-executable", default="mmseqs")
    parser.add_argument("--min-sequence-identity", type=float, default=0.3)
    parser.add_argument("--min-coverage", type=float, default=0.8)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--min-train-per-class", type=int, default=20)
    parser.add_argument("--min-validation-per-class", type=int, default=5)
    parser.add_argument("--min-test-per-class", type=int, default=5)
    args = parser.parse_args(argv)

    raw_records = load_annotation_records(
        args.protein_records, args.annotation_metadata
    )
    raw_records.extend(load_stability_records(args.stability_csv))
    records, quarantined = group_by_sequence(raw_records)
    if not records:
        raise ValueError("no valid ProteinCritic records")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    clustering = cluster_records(records, args.out_dir / "clustering", args)
    assignments = assign_clusters(
        records,
        seed=args.seed,
        required_task_keys=("stability_score",),
    )
    for record in records:
        record["split"] = assignments[record["protein_cluster"]]

    minimums = {
        "train": args.min_train_per_class,
        "validation": args.min_validation_per_class,
        "test": args.min_test_per_class,
    }
    pfam_labels = sorted(eligible_labels(records, "pfam_label", minimums))
    ec_labels = sorted(eligible_labels(records, "ec_label", minimums))
    pfam_vocab = {label: index for index, label in enumerate(pfam_labels)}
    ec_vocab = {str(label): index for index, label in enumerate(ec_labels)}
    ec_lookup = {label: index for index, label in enumerate(ec_labels)}
    if len(pfam_vocab) < 2 or len(ec_vocab) < 2:
        raise ValueError(
            "fewer than two supported Pfam or EC classes survive the held-out split"
        )

    records = [
        record
        for record in records
        if record.get("pfam_label") in pfam_vocab
        or record.get("ec_label") in ec_lookup
        or record.get("stability_score") is not None
    ]
    if not records:
        raise ValueError("no records retain a supported task target")

    report = split_report(records, ("pfam_label", "ec_label"))
    if report["cross_split_clusters"]:
        raise AssertionError("protein clusters cross train/validation/test splits")
    artifacts = {}
    for split in ("train", "validation", "test"):
        path = args.out_dir / f"{split}.jsonl"
        write_jsonl(
            path,
            [record for record in records if record["split"] == split],
            pfam_vocab,
            ec_lookup,
        )
        artifacts[split] = {"path": str(path.resolve()), "sha256": sha256(path)}
    vocab_path = args.out_dir / "task_vocabs.json"
    vocab_path.write_text(
        json.dumps(
            {
                "pfam": pfam_vocab,
                "ec": ec_vocab,
                "stability": {
                    "type": "regression",
                    "target": "deltaG",
                    "units": "kcal/mol",
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    artifacts["task_vocabs"] = {
        "path": str(vocab_path.resolve()),
        "sha256": sha256(vocab_path),
    }
    manifest = {
        "schema_version": 1,
        "protocol": "mmseqs_cluster_held_out_multitask_protein_critic",
        "seed": args.seed,
        "inputs": {
            "protein_records": {
                "path": str(args.protein_records.resolve()),
                "sha256": sha256(args.protein_records),
            },
            "annotation_metadata": {
                "path": str(args.annotation_metadata.resolve()),
                "sha256": sha256(args.annotation_metadata),
            },
            "stability_csv": {
                "path": str(args.stability_csv.resolve()),
                "sha256": sha256(args.stability_csv),
            },
        },
        "clustering": clustering,
        "retained_cluster_count": len(
            {record["protein_cluster"] for record in records}
        ),
        "exact_sequence_conflicts_quarantined": len(quarantined),
        "eligible_classes": {"pfam": pfam_labels, "ec_top_level": ec_labels},
        "minimum_class_support": minimums,
        "splits": report,
        "artifacts": artifacts,
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        f"[critic-data] records={len(records)} "
        f"retained_clusters={manifest['retained_cluster_count']} "
        f"source_clusters={clustering['cluster_count']} "
        f"pfam_classes={len(pfam_vocab)} ec_classes={len(ec_vocab)} out={args.out_dir}"
    )


if __name__ == "__main__":
    main()
