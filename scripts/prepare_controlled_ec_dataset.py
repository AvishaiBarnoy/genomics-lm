#!/usr/bin/env python3
"""Build an EC probe dataset aligned to a frozen CodonLM genome split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_ec_labels(path: Path) -> dict[str, int]:
    labels: dict[str, int] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            protein_id = str(row.get("ncbi_id", "")).strip()
            ec = str(row.get("ec", "")).strip()
            if protein_id and ec and ec[0].isdigit() and 1 <= int(ec[0]) <= 7:
                labels[protein_id] = int(ec[0])
    return labels


def _load_records(meta_path: Path, dna_path: Path, labels: dict[str, int]) -> list[dict]:
    dna = dna_path.read_text().splitlines()
    records = []
    with meta_path.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            protein_id = row.get("protein_id", "").strip()
            label = labels.get(protein_id)
            split = row.get("split", "").strip()
            if label is None or split not in {"train", "test"}:
                continue
            line_idx = int(row["line_idx"])
            if line_idx >= len(dna):
                raise ValueError(f"line_idx {line_idx} exceeds {dna_path}")
            sequence = dna[line_idx].strip().upper()
            protein = row.get("translation", "").strip().upper()
            if not sequence or not protein or "X" in protein or "*" in protein:
                continue
            records.append(
                {
                    "id": row["source_id"],
                    "protein_id": protein_id,
                    "genome": row["genome"],
                    "pretraining_split": split,
                    "dna": sequence,
                    "protein": protein,
                    "label": label,
                }
            )
    return records


def _cluster(records: list[dict], work_dir: Path, args) -> dict:
    executable = shutil.which(args.mmseqs_executable)
    if executable is None:
        raise RuntimeError(f"MMseqs2 executable not found: {args.mmseqs_executable}")
    work_dir.mkdir(parents=True, exist_ok=True)
    fasta = work_dir / "ec_proteins.fasta"
    with fasta.open("w") as handle:
        for index, record in enumerate(records):
            key = f"record-{index}"
            record["cluster_key"] = key
            handle.write(f">{key}\n{record['protein']}\n")
    prefix = work_dir / "clusters"
    command = [
        executable,
        "easy-cluster",
        str(fasta),
        str(prefix),
        str(work_dir / "tmp"),
        "--min-seq-id",
        str(args.min_protein_identity),
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
    cluster_path = Path(f"{prefix}_cluster.tsv")
    assignments = {}
    with cluster_path.open() as handle:
        for line in handle:
            representative, member = line.rstrip("\n").split("\t")[:2]
            assignments[member] = representative
    for record in records:
        try:
            record["protein_cluster"] = assignments[record["cluster_key"]]
        except KeyError as exc:
            raise RuntimeError(f"MMseqs2 omitted {record['cluster_key']}") from exc
    return {
        "tool": {
            "executable": executable,
            "version": (version.stdout or version.stderr).strip(),
        },
        "command": command,
        "thresholds": {
            "min_sequence_identity": args.min_protein_identity,
            "min_coverage": args.min_coverage,
            "coverage_mode": 0,
        },
        "protein_fasta_sha256": _sha256(fasta),
        "cluster_assignments_sha256": _sha256(cluster_path),
        "cluster_count": len(set(assignments.values())),
    }


def _write_csvs(out_dir: Path, split: str, records: list[dict]) -> None:
    with (out_dir / f"{split}_ec.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "label"])
        writer.writerows((record["id"], record["label"]) for record in records)
    with (out_dir / f"{split}_ec_seqs.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "seq"])
        writer.writerows((record["id"], record["dna"]) for record in records)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cds-meta", type=Path, required=True)
    parser.add_argument("--cds-dna", type=Path, required=True)
    parser.add_argument("--uniprot-metadata", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--mmseqs-executable", default="mmseqs")
    parser.add_argument("--min-protein-identity", type=float, default=0.3)
    parser.add_argument("--min-coverage", type=float, default=0.8)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--min-train-per-class", type=int, default=20)
    parser.add_argument("--min-test-per-class", type=int, default=5)
    args = parser.parse_args(argv)

    labels = _load_ec_labels(args.uniprot_metadata)
    records = _load_records(args.cds_meta, args.cds_dna, labels)
    if not records:
        raise ValueError("no EC-labelled records matched the frozen CDS metadata")
    clustering = _cluster(records, args.out_dir / "clustering", args)

    by_cluster: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_cluster[record["protein_cluster"]].append(record)
    crossing = {
        cluster
        for cluster, members in by_cluster.items()
        if {member["pretraining_split"] for member in members} == {"train", "test"}
    }

    train = [
        record
        for record in records
        if record["pretraining_split"] == "train" and record["protein_cluster"] not in crossing
    ]
    test = [
        record
        for record in records
        if record["pretraining_split"] == "test" and record["protein_cluster"] not in crossing
    ]
    train_counts, test_counts = Counter(r["label"] for r in train), Counter(
        r["label"] for r in test
    )
    eligible = {
        label
        for label in set(train_counts) & set(test_counts)
        if train_counts[label] >= args.min_train_per_class
        and test_counts[label] >= args.min_test_per_class
    }
    train = [record for record in train if record["label"] in eligible]
    test = [record for record in test if record["label"] in eligible]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if len(eligible) < 2:
        failure_report = {
            "protocol": "pretraining_genome_and_protein_cluster_held_out",
            "status": "failed",
            "reason": "fewer_than_two_eligible_ec_classes",
            "matched_records": len(records),
            "records_by_pretraining_split": dict(
                sorted(Counter(r["pretraining_split"] for r in records).items())
            ),
            "records_after_crossing_cluster_quarantine": {
                "train": sum(train_counts.values()),
                "test": sum(test_counts.values()),
            },
            "records_per_class_after_quarantine": {
                "train": dict(sorted(train_counts.items())),
                "test": dict(sorted(test_counts.items())),
            },
            "quarantined_cross_split_clusters": len(crossing),
            "thresholds": {
                "min_train_per_class": args.min_train_per_class,
                "min_test_per_class": args.min_test_per_class,
            },
            "clustering": clustering,
        }
        (args.out_dir / "split_report.json").write_text(
            json.dumps(failure_report, indent=2, sort_keys=True) + "\n"
        )
        raise ValueError(
            "fewer than two EC classes survive genome and protein-cluster holdouts"
        )
    if {r["protein_cluster"] for r in train} & {r["protein_cluster"] for r in test}:
        raise AssertionError("protein clusters overlap between train and test")

    _write_csvs(args.out_dir, "train", train)
    _write_csvs(args.out_dir, "test", test)
    with (args.out_dir / "split_assignments.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            ["id", "protein_id", "genome", "label", "protein_cluster", "split"]
        )
        for split, members in (("train", train), ("test", test)):
            for record in members:
                writer.writerow(
                    [
                        record["id"],
                        record["protein_id"],
                        record["genome"],
                        record["label"],
                        record["protein_cluster"],
                        split,
                    ]
                )
    report = {
        "protocol": "pretraining_genome_and_protein_cluster_held_out",
        "inputs": {
            "cds_meta": {"path": str(args.cds_meta.resolve()), "sha256": _sha256(args.cds_meta)},
            "cds_dna": {"path": str(args.cds_dna.resolve()), "sha256": _sha256(args.cds_dna)},
            "uniprot_metadata": {
                "path": str(args.uniprot_metadata.resolve()),
                "sha256": _sha256(args.uniprot_metadata),
            },
        },
        "clustering": clustering,
        "matched_records": len(records),
        "quarantined_cross_split_clusters": len(crossing),
        "eligible_classes": sorted(eligible),
        "train": {
            "records": len(train),
            "records_per_class": dict(sorted(Counter(r["label"] for r in train).items())),
            "genomes": sorted({r["genome"] for r in train}),
            "protein_clusters": len({r["protein_cluster"] for r in train}),
        },
        "test": {
            "records": len(test),
            "records_per_class": dict(sorted(Counter(r["label"] for r in test).items())),
            "genomes": sorted({r["genome"] for r in test}),
            "protein_clusters": len({r["protein_cluster"] for r in test}),
        },
        "cross_split_protein_clusters": 0,
    }
    (args.out_dir / "split_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"[ec] wrote {len(train)} train and {len(test)} test records "
        f"across {len(eligible)} classes"
    )


if __name__ == "__main__":
    main()
