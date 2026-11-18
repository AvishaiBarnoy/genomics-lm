import json
import pytest
from pathlib import Path
from scripts.audit_exact_duplicates import (
    audit_exact_duplicates,
    hash_sequence,
    normalize_dna,
    main,
    stream_dataset_meta_dna,
)


def test_fixture_deliberately_shared_sequence():
    """A fixture containing one deliberately shared sequence reports exactly that overlap."""
    records = [
        ("train_rec_1", "train", "ATGAAACCCGGGTAG"),
        ("train_rec_2", "train", "ATGGGGCCCAAATAA"),
        ("val_rec_1", "val", "ATGAAACCCGGGTAG"),  # Deliberately shared with train_rec_1
        ("val_rec_2", "val", "ATGTTTTTTTTTTAA"),
    ]

    report = audit_exact_duplicates(records)

    assert report["total_records_audited"] == 4
    assert report["unique_sequences"] == 3
    assert report["cross_split_duplicate_count"] == 1
    assert report["status"] == "failed"

    violation = report["cross_split_violations"][0]
    assert violation["splits"] == ["train", "val"]
    assert violation["records_by_split"]["train"] == ["train_rec_1"]
    assert violation["records_by_split"]["val"] == ["val_rec_1"]
    assert violation["sha256"] == hash_sequence("ATGAAACCCGGGTAG")


def test_normalization_invariance():
    """Lowercase bases, whitespace/newlines, and RNA uracil all normalize to identical hashes."""
    seq_canonical = "ATGCCCTAATAG"
    seq_lower = "atg cccta atag\n"
    seq_rna = "AUGCCCUAAUAG"

    h1 = hash_sequence(seq_canonical)
    h2 = hash_sequence(seq_lower)
    h3 = hash_sequence(seq_rna)

    assert h1 == h2 == h3

    records = [
        ("train_0", "train", seq_canonical),
        ("val_0", "val", seq_lower),
    ]
    report = audit_exact_duplicates(records)
    assert report["cross_split_duplicate_count"] == 1


def test_no_cross_split_duplicates():
    """Completely disjoint splits report zero cross-split duplicates."""
    records = [
        ("train_0", "train", "ATGAAA"),
        ("train_1", "train", "ATGCCC"),
        ("val_0", "val", "ATGGGG"),
        ("val_1", "val", "ATGTTT"),
    ]
    report = audit_exact_duplicates(records)
    assert report["cross_split_duplicate_count"] == 0
    assert report["status"] == "passed"
    assert report["cross_split_violations"] == []


def test_cli_separate_split_files(tmp_path):
    """CLI functions as expected with separate train.txt and val.txt files."""
    train_file = tmp_path / "train.txt"
    val_file = tmp_path / "val.txt"
    report_file = tmp_path / "report.json"

    train_file.write_text("ATGAAC\nATGCCC\n")
    val_file.write_text("ATGCCC\nATGTTT\n")  # ATGCCC is shared

    ret = main([
        "--train", str(train_file),
        "--val", str(val_file),
        "--report-json", str(report_file),
    ])

    assert ret == 0  # Diagnostic mode by default
    assert report_file.exists()

    data = json.loads(report_file.read_text())
    assert data["cross_split_duplicate_count"] == 1
    assert data["total_records_audited"] == 4
    assert data["cross_split_violations"][0]["records_by_split"]["train"] == ["train_1"]
    assert data["cross_split_violations"][0]["records_by_split"]["val"] == ["val_0"]


def test_cli_meta_dna_dataset(tmp_path):
    """CLI handles paired meta TSV and dna txt."""
    meta_file = tmp_path / "cds_meta.tsv"
    dna_file = tmp_path / "cds_dna.txt"
    report_file = tmp_path / "report.json"

    meta_file.write_text("source_id\tsplit\nrec_A\ttrain\nrec_B\tval\nrec_C\ttest\n")
    dna_file.write_text("ATGAAA\nATGAAA\nATGCCC\n")  # rec_A and rec_B share ATGAAA

    ret = main([
        "--meta", str(meta_file),
        "--dna", str(dna_file),
        "--report-json", str(report_file),
        "--fail-on-duplicate",
    ])

    assert ret == 1  # Should exit with 1 because --fail-on-duplicate is set
    data = json.loads(report_file.read_text())
    assert data["cross_split_duplicate_count"] == 1
    assert data["cross_split_violations"][0]["splits"] == ["train", "val"]


@pytest.mark.parametrize(
    ("metadata", "dna"),
    [
        ("source_id\tsplit\na\ttrain\nb\tval\n", "ATGAAA\n"),
        ("source_id\tsplit\na\ttrain\n", "ATGAAA\nATGCCC\n"),
    ],
)
def test_meta_dna_record_count_mismatch_fails(tmp_path, metadata, dna):
    meta_file = tmp_path / "cds_meta.tsv"
    dna_file = tmp_path / "cds_dna.txt"
    meta_file.write_text(metadata)
    dna_file.write_text(dna)

    with pytest.raises(ValueError, match="record counts differ"):
        list(stream_dataset_meta_dna(meta_file, dna_file))
