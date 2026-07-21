"""Preventive source-record leakage audits for scientific dataset preparation."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

from Bio.Seq import Seq


SPLIT_ORDER = {"train": 0, "val": 1, "test": 2}


class LeakageAuditError(RuntimeError):
    """Raised when a blocking leakage audit cannot pass."""


def normalize_cds(sequence: str) -> str:
    """Return the canonical DNA representation used for exact hashing."""
    return "".join(str(sequence).split()).upper().replace("U", "T")


def translate_cds(sequence: str, table: int = 11) -> str:
    """Translate a normalized CDS while retaining internal stop markers."""
    normalized = normalize_cds(sequence)
    usable = normalized[: len(normalized) - (len(normalized) % 3)]
    if not usable:
        return ""
    protein = str(Seq(usable).translate(table=table))
    if protein.endswith("*"):
        protein = protein[:-1]
    return protein.replace("*", "X")


def exact_cross_split_duplicates(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return full-CDS hashes whose source records occur in multiple splits."""
    by_hash: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        digest = hashlib.sha256(normalize_cds(record["sequence"]).encode("ascii")).hexdigest()
        by_hash[digest].append(record)

    violations = []
    for digest, members in sorted(by_hash.items()):
        splits = sorted({str(member["split"]) for member in members}, key=SPLIT_ORDER.get)
        if len(splits) < 2:
            continue
        violations.append(
            {
                "sha256": digest,
                "splits": splits,
                "source_ids": sorted(str(member["source_id"]) for member in members),
            }
        )
    return violations


def quarantine_cross_split_exact_duplicates(
    records: Sequence[Mapping[str, Any]],
    *,
    split_priority: Sequence[str] = ("test", "val", "train"),
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    """Keep cross-split duplicate families only in the highest-priority split."""
    priority = {split: index for index, split in enumerate(split_priority)}
    if set(priority) != set(SPLIT_ORDER):
        raise ValueError("split_priority must contain train, val, and test exactly once")

    by_hash: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        digest = hashlib.sha256(normalize_cds(record["sequence"]).encode("ascii")).hexdigest()
        by_hash[digest].append(record)

    removed_ids: set[int] = set()
    families = []
    removed_by_split = {split: 0 for split in SPLIT_ORDER}
    for digest, members in sorted(by_hash.items()):
        splits = {str(member["split"]) for member in members}
        if len(splits) < 2:
            continue
        kept_split = min(splits, key=priority.__getitem__)
        removed = [member for member in members if str(member["split"]) != kept_split]
        for member in removed:
            removed_ids.add(id(member))
            removed_by_split[str(member["split"])] += 1
        families.append(
            {
                "sha256": digest,
                "kept_split": kept_split,
                "kept_source_ids": sorted(
                    str(member["source_id"])
                    for member in members
                    if str(member["split"]) == kept_split
                ),
                "removed_source_ids": sorted(str(member["source_id"]) for member in removed),
            }
        )

    retained = [record for record in records if id(record) not in removed_ids]
    return retained, {
        "policy": "keep_highest_priority_split",
        "split_priority": list(split_priority),
        "duplicate_family_count": len(families),
        "removed_record_count": len(removed_ids),
        "removed_by_split": removed_by_split,
        "families": families,
    }


def cross_split_cluster_violations(
    clusters: Mapping[str, Sequence[str]],
    split_by_source: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Return protein clusters containing records from more than one split."""
    violations = []
    for representative, members in sorted(clusters.items()):
        source_ids = sorted(set(members))
        splits = sorted(
            {split_by_source[source_id] for source_id in source_ids},
            key=SPLIT_ORDER.get,
        )
        if len(splits) > 1:
            violations.append(
                {
                    "representative": representative,
                    "splits": splits,
                    "source_ids": source_ids,
                }
            )
    return violations


def _write_fasta(path: Path, records: Iterable[tuple[str, str]]) -> None:
    with path.open("w") as handle:
        for source_id, sequence in records:
            handle.write(f">{source_id}\n{sequence}\n")


def _run(command: list[str], commands: list[list[str]]) -> subprocess.CompletedProcess[str]:
    commands.append(command)
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        suffix = f": {detail}" if detail else ""
        raise LeakageAuditError(
            f"external audit command failed with exit code {exc.returncode}: "
            f"{' '.join(command)}{suffix}"
        ) from exc


def _parse_clusters(path: Path) -> dict[str, list[str]]:
    clusters: dict[str, list[str]] = defaultdict(list)
    with path.open() as handle:
        for line in handle:
            representative, member = line.rstrip("\n").split("\t")[:2]
            clusters[representative].append(member)
    return dict(clusters)


def _parse_nearest(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open() as handle:
        for line in handle:
            query, target, pident, alnlen, qlen, tlen = line.rstrip("\n").split("\t")
            rows.append(
                {
                    "query_id": query,
                    "target_id": target,
                    "identity": float(pident) / 100.0,
                    "alignment_length": int(alnlen),
                    "query_length": int(qlen),
                    "target_length": int(tlen),
                    "query_coverage": int(alnlen) / max(1, int(qlen)),
                }
            )
    return rows


def _parse_minimap_paf(path: Path) -> list[dict[str, Any]]:
    """Return the best primary nucleotide alignment reported for each query."""
    best: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return []
    with path.open() as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 12:
                continue
            query, qlen = fields[0], int(fields[1])
            target, tlen = fields[5], int(fields[6])
            matches, alnlen, mapq = int(fields[9]), int(fields[10]), int(fields[11])
            row = {
                "query_id": query,
                "target_id": target,
                "identity": matches / max(1, alnlen),
                "alignment_length": alnlen,
                "query_length": qlen,
                "target_length": tlen,
                "query_coverage": alnlen / max(1, qlen),
                "mapq": mapq,
                "matching_bases": matches,
            }
            score = (matches, alnlen, mapq, target)
            previous = best.get(query)
            if previous is None or score > previous["_score"]:
                row["_score"] = score
                best[query] = row
    rows = []
    for query in sorted(best):
        row = best[query]
        row.pop("_score", None)
        rows.append(row)
    return rows


def matching_substring_coverage(
    sequence: str, training_sequences: Sequence[str], window_size: int
) -> float:
    """Return the fraction of query positions covered by exact training windows."""
    if window_size < 1:
        raise ValueError("window_size must be at least 1")
    if len(sequence) < window_size:
        return 0.0
    training_windows = {
        training[start : start + window_size]
        for training in training_sequences
        for start in range(max(0, len(training) - window_size + 1))
    }
    return _coverage_from_index(sequence, training_windows, window_size)


def _coverage_from_index(
    sequence: str, training_windows: set[str], window_size: int
) -> float:
    if len(sequence) < window_size or not training_windows:
        return 0.0
    covered = bytearray(len(sequence))
    for start in range(len(sequence) - window_size + 1):
        if sequence[start : start + window_size] in training_windows:
            covered[start : start + window_size] = b"\x01" * window_size
    return sum(covered) / len(sequence)


def _identity_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    identities = sorted(float(row["identity"]) for row in rows)
    if not identities:
        return {
            "count": 0,
            "min": None,
            "median": None,
            "p90": None,
            "p95": None,
            "max": None,
        }

    def percentile(fraction: float) -> float:
        index = fraction * (len(identities) - 1)
        lower = int(index)
        upper = min(lower + 1, len(identities) - 1)
        weight = index - lower
        return identities[lower] * (1.0 - weight) + identities[upper] * weight

    return {
        "count": len(identities),
        "min": identities[0],
        "median": median(identities),
        "p90": percentile(0.9),
        "p95": percentile(0.95),
        "max": identities[-1],
    }


def run_mmseqs_audit(
    records: Sequence[Mapping[str, Any]],
    work_dir: Path,
    *,
    min_protein_identity: float,
    min_coverage: float,
    threads: int = 1,
    executable: str = "mmseqs",
    nucleotide_executable: str = "minimap2",
    nucleotide_preset: str = "asm20",
    nearest_query_batch_size: int = 4096,
    split_memory_limit: str = "0",
) -> dict[str, Any]:
    """Cluster translated CDS records and find held-out nearest neighbors."""
    if nearest_query_batch_size < 1:
        raise ValueError("nearest_query_batch_size must be at least 1")
    resolved = shutil.which(executable)
    if resolved is None:
        raise LeakageAuditError(
            f"MMseqs2 executable {executable!r} was not found; scientific preparation requires the protein-homology audit"
        )
    resolved_nucleotide = shutil.which(nucleotide_executable)
    if resolved_nucleotide is None:
        raise LeakageAuditError(
            f"nucleotide aligner {nucleotide_executable!r} was not found; "
            "scientific preparation requires the nucleotide nearest-neighbor audit"
        )
    work_dir.mkdir(parents=True, exist_ok=True)
    commands: list[list[str]] = []
    version_result = _run([resolved, "version"], commands)
    version = (version_result.stdout or version_result.stderr).strip()
    nucleotide_version_result = _run([resolved_nucleotide, "--version"], commands)
    nucleotide_version = (
        nucleotide_version_result.stdout or nucleotide_version_result.stderr
    ).strip()

    proteins = [(str(record["source_id"]), translate_cds(record["sequence"])) for record in records]
    proteins = [(source_id, sequence) for source_id, sequence in proteins if sequence]
    protein_fasta = work_dir / "all_proteins.fasta"
    _write_fasta(protein_fasta, proteins)
    cluster_prefix = work_dir / "protein_clusters"
    cluster_tmp = work_dir / "cluster_tmp"
    _run(
        [
            resolved,
            "easy-cluster",
            str(protein_fasta),
            str(cluster_prefix),
            str(cluster_tmp),
            "--min-seq-id",
            str(min_protein_identity),
            "-c",
            str(min_coverage),
            "--cov-mode",
            "0",
            "--cluster-mode",
            "0",
            "--threads",
            str(threads),
        ],
        commands,
    )
    clusters = _parse_clusters(Path(f"{cluster_prefix}_cluster.tsv"))

    train = [record for record in records if record["split"] == "train"]
    held_out = [record for record in records if record["split"] in {"val", "test"}]
    nearest: dict[str, Any] = {}
    for sequence_type in ("nucleotide", "protein"):
        if sequence_type == "protein":
            convert = translate_cds
        else:
            convert = normalize_cds
        train_fasta = work_dir / f"train_{sequence_type}.fasta"
        _write_fasta(train_fasta, ((str(r["source_id"]), convert(r["sequence"])) for r in train))
        output = work_dir / f"nearest_{sequence_type}.tsv"
        if sequence_type == "nucleotide":
            query_fasta = work_dir / "held_out_nucleotide.fasta"
            _write_fasta(
                query_fasta,
                ((str(record["source_id"]), convert(record["sequence"])) for record in held_out),
            )
            result = _run(
                [
                    resolved_nucleotide,
                    "-x",
                    nucleotide_preset,
                    "--secondary=no",
                    "-t",
                    str(threads),
                    str(train_fasta),
                    str(query_fasta),
                ],
                commands,
            )
            output = work_dir / "nearest_nucleotide.paf"
            output.write_text(result.stdout)
            rows = _parse_minimap_paf(output)
            nearest[sequence_type] = {
                "artifact": str(output),
                "tool": {
                    "name": "Minimap2",
                    "executable": resolved_nucleotide,
                    "version": nucleotide_version,
                    "preset": nucleotide_preset,
                },
                "query_batch_count": 1,
                "query_count": len(held_out),
                "hit_count": len(rows),
                "hit_fraction": len(rows) / len(held_out) if held_out else 0.0,
                "summary": _identity_summary(rows),
            }
            continue
        output.write_text("")
        rows = []
        converted_held_out = [
            (str(record["source_id"]), convert(record["sequence"]))
            for record in held_out
        ]
        for batch_index, start in enumerate(
            range(0, len(converted_held_out), nearest_query_batch_size)
        ):
            query_fasta = work_dir / f"held_out_{sequence_type}_{batch_index:04d}.fasta"
            part_output = work_dir / f"nearest_{sequence_type}_{batch_index:04d}.tsv"
            search_tmp = work_dir / f"search_{sequence_type}_{batch_index:04d}_tmp"
            _write_fasta(
                query_fasta,
                converted_held_out[start : start + nearest_query_batch_size],
            )
            _run(
                [
                    resolved,
                    "easy-search",
                    str(query_fasta),
                    str(train_fasta),
                    str(part_output),
                    str(search_tmp),
                    "--format-output",
                    "query,target,pident,alnlen,qlen,tlen",
                    "--max-seqs",
                    "1",
                    "--search-type",
                    "1",
                    "--split-memory-limit",
                    split_memory_limit,
                    "--threads",
                    str(threads),
                ],
                commands,
            )
            part_text = part_output.read_text() if part_output.exists() else ""
            with output.open("a") as handle:
                handle.write(part_text)
            rows.extend(_parse_nearest(part_output))
        nearest[sequence_type] = {
            "artifact": str(output),
            "query_batch_count": (
                len(converted_held_out) + nearest_query_batch_size - 1
            )
            // nearest_query_batch_size,
            "query_count": len(converted_held_out),
            "hit_count": len(rows),
            "hit_fraction": (
                len(rows) / len(converted_held_out) if converted_held_out else 0.0
            ),
            "summary": _identity_summary(rows),
        }

    cluster_artifact = Path(f"{cluster_prefix}_cluster.tsv")
    retained_artifacts = {
        cluster_artifact.resolve(),
        *(Path(result["artifact"]).resolve() for result in nearest.values()),
    }
    for path in work_dir.iterdir():
        if path.resolve() in retained_artifacts:
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

    return {
        "tool": {"name": "MMseqs2", "executable": resolved, "version": version},
        "nucleotide_tool": {
            "name": "Minimap2",
            "executable": resolved_nucleotide,
            "version": nucleotide_version,
            "preset": nucleotide_preset,
        },
        "parameters": {
            "min_protein_identity": min_protein_identity,
            "min_coverage": min_coverage,
            "cov_mode": 0,
            "cluster_mode": 0,
            "threads": threads,
            "nearest_query_batch_size": nearest_query_batch_size,
            "split_memory_limit": split_memory_limit,
        },
        "commands": commands,
        "cluster_artifact": str(cluster_artifact),
        "intermediates_cleaned": True,
        "_clusters": clusters,
        "nearest_neighbors": nearest,
    }


def audit_source_records(
    records: Sequence[Mapping[str, Any]],
    output_path: Path,
    *,
    min_protein_identity: float = 0.3,
    min_coverage: float = 0.8,
    threads: int = 1,
    executable: str = "mmseqs",
    skip_homology: bool = False,
    allow_exact_duplicates: bool = False,
    protein_homology_policy: str = "block",
    nucleotide_executable: str = "minimap2",
    nucleotide_preset: str = "asm20",
    nearest_query_batch_size: int = 4096,
    split_memory_limit: str = "0",
) -> dict[str, Any]:
    """Run blocking exact and protein-homology audits and always write JSON."""
    if protein_homology_policy not in {"block", "report"}:
        raise ValueError("protein_homology_policy must be 'block' or 'report'")
    exact = exact_cross_split_duplicates(records)
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "pending",
        "record_count": len(records),
        "thresholds": {
            "max_exact_cross_split_duplicates": 0,
            "max_cross_split_protein_clusters": (
                0 if protein_homology_policy == "block" else None
            ),
            "min_protein_identity": min_protein_identity,
            "min_coverage": min_coverage,
        },
        "exact_duplicates": {"count": len(exact), "violations": exact},
        "homology_audit_skipped": skip_homology,
        "exact_duplicate_override": allow_exact_duplicates,
        "protein_homology_policy": protein_homology_policy,
    }
    blocking_reasons = []
    if exact and not allow_exact_duplicates:
        blocking_reasons.append("cross_split_exact_duplicates")

    try:
        if not skip_homology:
            mmseqs = run_mmseqs_audit(
                records,
                output_path.parent / "leakage_audit_work",
                min_protein_identity=min_protein_identity,
                min_coverage=min_coverage,
                threads=threads,
                executable=executable,
                nucleotide_executable=nucleotide_executable,
                nucleotide_preset=nucleotide_preset,
                nearest_query_batch_size=nearest_query_batch_size,
                split_memory_limit=split_memory_limit,
            )
            split_by_source = {str(record["source_id"]): str(record["split"]) for record in records}
            clusters = mmseqs.pop("_clusters")
            protein_violations = cross_split_cluster_violations(clusters, split_by_source)
            mmseqs["cluster_count"] = len(clusters)
            mmseqs["cross_split_cluster_count"] = len(protein_violations)
            mmseqs["cross_split_violations"] = protein_violations
            report["protein_homology"] = mmseqs
            if protein_violations and protein_homology_policy == "block":
                blocking_reasons.append("cross_split_protein_clusters")
        else:
            report["protein_homology"] = None
    except (LeakageAuditError, subprocess.CalledProcessError, OSError, ValueError) as exc:
        report["status"] = "error"
        report["error"] = str(exc)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n")
        raise LeakageAuditError(str(exc)) from exc

    report["blocking_reasons"] = blocking_reasons
    report["status"] = "failed" if blocking_reasons else "passed"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    if blocking_reasons:
        raise LeakageAuditError(
            "Leakage audit failed: " + ", ".join(blocking_reasons)
        )
    return report


def audit_generated_sequences(
    training: Sequence[Mapping[str, Any]],
    generated: Sequence[Mapping[str, Any]],
    output_path: Path,
    *,
    nucleotide_window: int = 30,
    protein_window: int = 10,
    threads: int = 1,
    executable: str = "mmseqs",
) -> dict[str, Any]:
    """Report nearest training identities and matching-substring coverage."""
    resolved = shutil.which(executable)
    if resolved is None:
        raise LeakageAuditError(f"MMseqs2 executable {executable!r} was not found")
    work_dir = output_path.parent / f"{output_path.stem}_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    commands: list[list[str]] = []
    version_result = _run([resolved, "version"], commands)
    version = (version_result.stdout or version_result.stderr).strip()
    nearest_by_type: dict[str, dict[str, dict[str, Any]]] = {}

    converters = {"nucleotide": normalize_cds, "protein": translate_cds}
    for sequence_type, convert in converters.items():
        train_fasta = work_dir / f"train_{sequence_type}.fasta"
        generated_fasta = work_dir / f"generated_{sequence_type}.fasta"
        _write_fasta(
            train_fasta,
            ((str(record["source_id"]), convert(record["sequence"])) for record in training),
        )
        _write_fasta(
            generated_fasta,
            ((str(record["source_id"]), convert(record["sequence"])) for record in generated),
        )
        output = work_dir / f"nearest_{sequence_type}.tsv"
        _run(
            [
                resolved,
                "easy-search",
                str(generated_fasta),
                str(train_fasta),
                str(output),
                str(work_dir / f"search_{sequence_type}_tmp"),
                "--format-output",
                "query,target,pident,alnlen,qlen,tlen",
                "--max-seqs",
                "1",
                "--search-type",
                "1" if sequence_type == "protein" else "3",
                "--threads",
                str(threads),
            ],
            commands,
        )
        nearest_by_type[sequence_type] = {
            row["query_id"]: row for row in _parse_nearest(output)
        }

    training_dna = [normalize_cds(record["sequence"]) for record in training]
    training_proteins = [translate_cds(record["sequence"]) for record in training]
    nucleotide_windows = {
        sequence[start : start + nucleotide_window]
        for sequence in training_dna
        for start in range(max(0, len(sequence) - nucleotide_window + 1))
    }
    protein_windows = {
        sequence[start : start + protein_window]
        for sequence in training_proteins
        for start in range(max(0, len(sequence) - protein_window + 1))
    }
    rows = []
    for record in generated:
        source_id = str(record["source_id"])
        dna = normalize_cds(record["sequence"])
        protein = translate_cds(record["sequence"])
        rows.append(
            {
                "source_id": source_id,
                "nucleotide_nearest": nearest_by_type["nucleotide"].get(source_id),
                "protein_nearest": nearest_by_type["protein"].get(source_id),
                "nucleotide_training_match_coverage": _coverage_from_index(
                    dna, nucleotide_windows, nucleotide_window
                ),
                "protein_training_match_coverage": _coverage_from_index(
                    protein, protein_windows, protein_window
                ),
            }
        )

    report = {
        "schema_version": 1,
        "tool": {"name": "MMseqs2", "executable": resolved, "version": version},
        "parameters": {
            "nucleotide_window": nucleotide_window,
            "protein_window": protein_window,
            "threads": threads,
        },
        "commands": commands,
        "training_record_count": len(training),
        "generated_record_count": len(generated),
        "records": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


__all__ = [
    "LeakageAuditError",
    "audit_generated_sequences",
    "audit_source_records",
    "cross_split_cluster_violations",
    "exact_cross_split_duplicates",
    "quarantine_cross_split_exact_duplicates",
    "normalize_cds",
    "matching_substring_coverage",
    "run_mmseqs_audit",
    "translate_cds",
]
