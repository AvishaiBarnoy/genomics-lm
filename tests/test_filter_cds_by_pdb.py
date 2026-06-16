import csv

import pytest

from scripts.filter_cds_by_pdb import (
    filter_by_line_indices,
    load_records,
    read_line_indices,
    validate_join_keys,
    write_subset,
)


def test_line_index_filter_preserves_source_indices(tmp_path):
    dna = tmp_path / "cds_dna.txt"
    meta = tmp_path / "cds_meta.tsv"
    indices = tmp_path / "indices.txt"
    out_dir = tmp_path / "out"

    dna.write_text("ATGAAA\nATGCCC\nATGGGG\n")
    meta.write_text(
        "line_idx\tgenome\tprotein_id\n"
        "0\tg1\tp0\n"
        "1\tg1\tp1\n"
        "2\tg2\tp2\n"
    )
    indices.write_text("# structured rows\n1\n2\n")

    header, records = load_records(dna, meta)
    selected = filter_by_line_indices(records, read_line_indices(indices))
    write_subset(selected, header, out_dir, {"mode": "line_indices"})

    assert (out_dir / "cds_dna.txt").read_text().splitlines() == ["ATGCCC", "ATGGGG"]

    with (out_dir / "cds_meta.tsv").open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert rows[0]["line_idx"] == "0"
    assert rows[0]["source_line_idx"] == "1"
    assert rows[0]["protein_id"] == "p1"
    assert rows[1]["line_idx"] == "1"
    assert rows[1]["source_line_idx"] == "2"


def test_missing_line_index_raises(tmp_path):
    dna = tmp_path / "cds_dna.txt"
    meta = tmp_path / "cds_meta.tsv"
    dna.write_text("ATGAAA\n")
    meta.write_text("line_idx\tgenome\n0\tg1\n")

    _, records = load_records(dna, meta)
    with pytest.raises(ValueError, match="requested line indices"):
        filter_by_line_indices(records, {0, 9})


def test_join_validation_requires_enriched_cds_metadata():
    with pytest.raises(ValueError, match="CDS metadata has no protein/gene join key"):
        validate_join_keys(
            meta_header=["line_idx", "genome"],
            uniprot_header=["Entry", "Gene Names", "Sequence"],
        )
