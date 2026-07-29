from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from src.codonlm.leakage_audit import (
    LeakageAuditError,
    audit_generated_sequences,
    audit_source_records,
    cross_split_cluster_violations,
    exact_cross_split_duplicates,
    matching_substring_coverage,
    quarantine_cross_split_exact_duplicates,
    translate_cds,
)


def _record(source_id: str, split: str, sequence: str) -> dict[str, str]:
    return {"source_id": source_id, "split": split, "sequence": sequence}


def test_exact_duplicate_hashes_normalized_full_cds():
    records = [
        _record("train-a", "train", "atg gcc taa"),
        _record("test-a", "test", "ATGGCCTAA"),
        _record("val-unique", "val", "ATGGCTTAA"),
    ]

    violations = exact_cross_split_duplicates(records)

    assert len(violations) == 1
    assert violations[0]["splits"] == ["train", "test"]
    assert violations[0]["source_ids"] == ["test-a", "train-a"]


def test_translation_removes_terminal_stop_for_mmseqs():
    assert translate_cds("ATGGCTTAA") == "MA"


def test_exact_duplicate_quarantine_preserves_heldout_priority():
    records = [
        _record("train-a", "train", "ATGGCCTAA"),
        _record("train-b", "train", "ATGGCCTAA"),
        _record("val-a", "val", "ATGGCCTAA"),
        _record("test-a", "test", "ATGGCCTAA"),
        _record("train-unique", "train", "ATGGCTTAA"),
    ]

    retained, report = quarantine_cross_split_exact_duplicates(records)

    assert {record["source_id"] for record in retained} == {"test-a", "train-unique"}
    assert report["duplicate_family_count"] == 1
    assert report["removed_record_count"] == 3
    assert report["removed_by_split"] == {"train": 2, "val": 1, "test": 0}
    assert exact_cross_split_duplicates(retained) == []


def test_matching_substring_coverage_marks_query_positions():
    assert matching_substring_coverage("AAACCCGGG", ["TTTAAACCC"], 3) == 6 / 9


def test_protein_cluster_threshold_behavior_is_cross_split_only():
    clusters = {
        "train-a": ["train-a", "train-b"],
        "train-c": ["train-c", "test-a"],
        "val-a": ["val-a"],
    }
    splits = {
        "train-a": "train",
        "train-b": "train",
        "train-c": "train",
        "test-a": "test",
        "val-a": "val",
    }

    violations = cross_split_cluster_violations(clusters, splits)

    assert violations == [
        {
            "representative": "train-c",
            "splits": ["train", "test"],
            "source_ids": ["test-a", "train-c"],
        }
    ]


def test_exact_duplicate_blocks_even_when_homology_is_explicitly_skipped(tmp_path):
    report_path = tmp_path / "leakage.json"
    records = [
        _record("train-a", "train", "ATGGCCTAA"),
        _record("test-a", "test", "ATGGCCTAA"),
    ]

    with pytest.raises(LeakageAuditError, match="cross_split_exact_duplicates"):
        audit_source_records(records, report_path, skip_homology=True)

    report = json.loads(report_path.read_text())
    assert report["status"] == "failed"
    assert report["exact_duplicates"]["count"] == 1
    assert report["blocking_reasons"] == ["cross_split_exact_duplicates"]


def _write_fake_mmseqs(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env python3
import sys
from pathlib import Path

def fasta(path):
    records = []
    name = None
    sequence = []
    for line in Path(path).read_text().splitlines():
        if line.startswith('>'):
            if name is not None:
                records.append((name, ''.join(sequence)))
            name = line[1:]
            sequence = []
        else:
            sequence.append(line.strip())
    if name is not None:
        records.append((name, ''.join(sequence)))
    return records

command = sys.argv[1]
if command == 'version':
    print('fake-mmseqs-1.0')
elif command == 'easy-cluster':
    groups = {}
    for name, sequence in fasta(sys.argv[2]):
        groups.setdefault(sequence, []).append(name)
    output = Path(sys.argv[3] + '_cluster.tsv')
    output.write_text(''.join(
        f'{members[0]}\\t{member}\\n'
        for members in groups.values()
        for member in members
    ))
elif command == 'easy-search':
    queries = fasta(sys.argv[2])
    targets = fasta(sys.argv[3])
    rows = []
    for query_id, query in queries:
        if not targets:
            continue
        target_id, target = max(
            targets,
            key=lambda item: sum(a == b for a, b in zip(query, item[1])),
        )
        length = min(len(query), len(target))
        identity = 100.0 * sum(a == b for a, b in zip(query, target)) / max(1, length)
        rows.append(f'{query_id}\\t{target_id}\\t{identity}\\t{length}\\t{len(query)}\\t{len(target)}\\n')
    Path(sys.argv[4]).write_text(''.join(rows))
else:
    raise SystemExit(f'unsupported command: {command}')
"""
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _write_fake_minimap2(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env python3
import sys
from pathlib import Path

def fasta(path):
    records = []
    name = None
    sequence = []
    for line in Path(path).read_text().splitlines():
        if line.startswith('>'):
            if name is not None:
                records.append((name, ''.join(sequence)))
            name = line[1:]
            sequence = []
        else:
            sequence.append(line.strip())
    if name is not None:
        records.append((name, ''.join(sequence)))
    return records

if '--version' in sys.argv:
    print('fake-minimap2-1.0')
else:
    targets = fasta(sys.argv[-2])
    queries = fasta(sys.argv[-1])
    if targets:
        target_id, target = targets[0]
        for query_id, query in queries:
            length = min(len(query), len(target))
            matches = sum(a == b for a, b in zip(query, target))
            print(
                f'{query_id}\t{len(query)}\t0\t{length}\t+\t{target_id}\t'
                f'{len(target)}\t0\t{length}\t{matches}\t{length}\t60\ttp:A:P'
            )
"""
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def test_mmseqs_protein_cluster_violation_blocks_and_records_provenance(tmp_path):
    executable = tmp_path / "mmseqs"
    nucleotide_executable = tmp_path / "minimap2"
    _write_fake_mmseqs(executable)
    _write_fake_minimap2(nucleotide_executable)
    report_path = tmp_path / "leakage.json"
    records = [
        _record("train-synonym", "train", "ATGGCTGCTTAA"),
        _record("test-synonym", "test", "ATGGCCGCCTAA"),
        _record("val-unique", "val", "ATGAAACAATAA"),
    ]

    with pytest.raises(LeakageAuditError, match="cross_split_protein_clusters"):
        audit_source_records(
            records,
            report_path,
            executable=str(executable),
            nucleotide_executable=str(nucleotide_executable),
            min_protein_identity=0.3,
            min_coverage=0.8,
        )

    report = json.loads(report_path.read_text())
    homology = report["protein_homology"]
    assert report["status"] == "failed"
    assert homology["tool"]["version"] == "fake-mmseqs-1.0"
    assert homology["parameters"]["min_protein_identity"] == 0.3
    assert homology["cross_split_cluster_count"] == 1
    assert homology["cross_split_violations"][0]["source_ids"] == [
        "test-synonym",
        "train-synonym",
    ]
    assert homology["nearest_neighbors"]["protein"]["summary"]["count"] == 2
    search_commands = [command for command in homology["commands"] if command[1] == "easy-search"]
    assert [command[command.index("--search-type") + 1] for command in search_commands] == ["1"]
    assert homology["nearest_neighbors"]["nucleotide"]["tool"]["version"] == (
        "fake-minimap2-1.0"
    )
    assert homology["intermediates_cleaned"] is True
    work_dir = tmp_path / "leakage_audit_work"
    assert sorted(path.name for path in work_dir.iterdir()) == [
        "nearest_nucleotide.paf",
        "nearest_protein.tsv",
        "protein_clusters_cluster.tsv",
    ]


def test_mmseqs_protein_cluster_report_policy_is_nonblocking(tmp_path):
    executable = tmp_path / "mmseqs"
    nucleotide_executable = tmp_path / "minimap2"
    _write_fake_mmseqs(executable)
    _write_fake_minimap2(nucleotide_executable)
    report_path = tmp_path / "leakage.json"
    records = [
        _record("train-synonym", "train", "ATGGCTGCTTAA"),
        _record("test-synonym", "test", "ATGGCCGCCTAA"),
        _record("val-unique", "val", "ATGAAACAATAA"),
    ]

    report = audit_source_records(
        records,
        report_path,
        executable=str(executable),
        nucleotide_executable=str(nucleotide_executable),
        protein_homology_policy="report",
    )

    assert report["status"] == "passed"
    assert report["blocking_reasons"] == []
    assert report["protein_homology_policy"] == "report"
    assert report["thresholds"]["max_cross_split_protein_clusters"] is None
    assert report["protein_homology"]["cross_split_cluster_count"] == 1


def test_generated_audit_reports_nearest_identity_and_match_coverage(tmp_path):
    executable = tmp_path / "mmseqs"
    nucleotide_executable = tmp_path / "minimap2"
    _write_fake_mmseqs(executable)
    _write_fake_minimap2(nucleotide_executable)
    output = tmp_path / "generated_audit.json"

    report = audit_generated_sequences(
        [_record("train-a", "train", "ATGGCTGCTGCTTAA")],
        [_record("generated-a", "generated", "ATGGCTGCTAAATAA")],
        output,
        nucleotide_window=6,
        protein_window=2,
        executable=str(executable),
        nucleotide_executable=str(nucleotide_executable),
    )

    record = report["records"][0]
    assert output.exists()
    assert report["tool"]["version"] == "fake-mmseqs-1.0"
    assert record["nucleotide_nearest"]["target_id"] == "train-a"
    assert record["protein_nearest"]["target_id"] == "train-a"
    assert 0.0 < record["nucleotide_training_match_coverage"] <= 1.0
    assert 0.0 < record["protein_training_match_coverage"] <= 1.0


def test_generated_audit_batches_training_and_keeps_global_best(tmp_path):
    executable = tmp_path / "mmseqs"
    nucleotide_executable = tmp_path / "minimap2"
    _write_fake_mmseqs(executable)
    _write_fake_minimap2(nucleotide_executable)
    output = tmp_path / "generated_audit.json"

    report = audit_generated_sequences(
        [
            _record("train-poor", "train", "ATGTTTTTTTTTTAA"),
            _record("train-best", "train", "ATGGCTGCTAAATAA"),
        ],
        [_record("generated-a", "generated", "ATGGCTGCTAAATAA")],
        output,
        nucleotide_window=6,
        protein_window=2,
        executable=str(executable),
        nucleotide_executable=str(nucleotide_executable),
        training_batch_size=1,
    )

    record = report["records"][0]
    assert record["nucleotide_nearest"] is not None
    assert record["protein_nearest"]["target_id"] == "train-best"
    assert report["parameters"]["training_batch_size"] == 1
    assert sum(command[1] == "easy-search" for command in report["commands"]) == 2
