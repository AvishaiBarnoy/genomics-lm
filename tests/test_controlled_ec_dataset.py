import csv
import json

import pytest

from scripts.audit_downstream_pretraining import main as audit_downstream
from scripts.prepare_controlled_ec_dataset import main
from src.codonlm.leakage_audit import LeakageAuditError
from tests.test_leakage_audit import _write_fake_minimap2, _write_fake_mmseqs


def test_controlled_ec_split_respects_pretraining_and_protein_clusters(tmp_path):
    metadata = tmp_path / "cds_meta.tsv"
    dna = tmp_path / "cds_dna.txt"
    uniprot = tmp_path / "uniprot.csv"
    rows = [
        ("train-a1", "p1", "g-train", "train", "MAAAAA", "ATG" + "GCT" * 5, 1),
        ("test-a1", "p2", "g-test", "test", "MAAAAA", "ATG" + "GCT" * 5, 1),
        ("train-a2", "p3", "g-train", "train", "MCCCCC", "ATG" + "TGT" * 5, 1),
        ("test-a2", "p4", "g-test", "test", "MDDDDD", "ATG" + "GAT" * 5, 1),
        ("train-b1", "p5", "g-train", "train", "MEEEEE", "ATG" + "GAA" * 5, 2),
        ("test-b1", "p6", "g-test", "test", "MFFFFF", "ATG" + "TTT" * 5, 2),
    ]
    with metadata.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            ["line_idx", "split", "source_id", "genome", "protein_id", "translation"]
        )
        for index, row in enumerate(rows):
            writer.writerow([index, row[3], row[0], row[2], row[1], row[4]])
    dna.write_text("\n".join(row[5] for row in rows) + "\n")
    with uniprot.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ncbi_id", "ec"])
        for row in rows:
            writer.writerow([row[1], f"{row[6]}.1.1.1"])

    fake_mmseqs = tmp_path / "mmseqs"
    _write_fake_mmseqs(fake_mmseqs)
    out = tmp_path / "controlled"
    main(
        [
            "--cds-meta",
            str(metadata),
            "--cds-dna",
            str(dna),
            "--uniprot-metadata",
            str(uniprot),
            "--out-dir",
            str(out),
            "--mmseqs-executable",
            str(fake_mmseqs),
            "--min-train-per-class",
            "1",
            "--min-test-per-class",
            "1",
        ]
    )

    report = json.loads((out / "split_report.json").read_text())
    assert report["protocol"] == "pretraining_genome_and_protein_cluster_held_out"
    assert report["cross_split_protein_clusters"] == 0
    assert report["quarantined_cross_split_clusters"] == 1
    assert report["train"]["genomes"] == ["g-train"]
    assert report["test"]["genomes"] == ["g-test"]
    with (out / "train_ec.csv").open() as handle:
        train_ids = {row["id"] for row in csv.DictReader(handle)}
    with (out / "test_ec.csv").open() as handle:
        test_ids = {row["id"] for row in csv.DictReader(handle)}
    assert "train-a1" not in train_ids
    assert "test-a1" not in test_ids


def test_downstream_audit_blocks_exact_pretraining_duplicates(tmp_path):
    metadata = tmp_path / "cds_meta.tsv"
    metadata.write_text("line_idx\tsplit\tsource_id\n0\ttrain\ttrain-a\n")
    dna = tmp_path / "cds_dna.txt"
    dna.write_text("ATGGCTGCTTAA\n")
    downstream = tmp_path / "test.csv"
    downstream.write_text("id,seq\ntest-a,ATGGCTGCTTAA\n")
    mmseqs, minimap2 = tmp_path / "mmseqs", tmp_path / "minimap2"
    _write_fake_mmseqs(mmseqs)
    _write_fake_minimap2(minimap2)
    output = tmp_path / "audit.json"

    with pytest.raises(LeakageAuditError, match="cross_split_exact_duplicates"):
        audit_downstream(
            [
                "--cds-meta",
                str(metadata),
                "--cds-dna",
                str(dna),
                "--downstream-seqs",
                str(downstream),
                "--output",
                str(output),
                "--mmseqs-executable",
                str(mmseqs),
                "--minimap2-executable",
                str(minimap2),
            ]
        )

    report = json.loads(output.read_text())
    assert report["status"] == "failed"
    assert report["exact_duplicates"]["count"] == 1
