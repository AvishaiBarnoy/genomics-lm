from src.codonlm import codon_tokenize as ct
import csv
import subprocess
import pytest


def test_to_ids_basic():
    # ATG (start/M), TTT (F)
    dna = "ATGTTT"
    ids = ct.to_ids(dna)
    # Expect: <bos>, ATG, TTT, <eog>
    assert ids[0] == ct.stoi["<bos>"]
    assert ids[-1] == ct.stoi["<eog>"]
    assert ct.VOCAB[ids[1]] == "ATG"
    assert ct.VOCAB[ids[2]] == "TTT"


def _tokens(fragment):
    return [ct.itos[token_id] for token_id in fragment.ids]


def test_to_ids_rejects_ambiguous_codons():
    with pytest.raises(ct.AmbiguousCodonError, match="codon index 1"):
        ct.to_ids("ATGNNNGCT")
    with pytest.raises(ct.AmbiguousCodonError, match="partial codon"):
        ct.to_ids("ATGNN")


def test_fragment_tokenizer_preserves_internal_ambiguity_boundary():
    result = ct.tokenize_cds_fragments(
        "ATGNNNGCTGAA", source_id="gene-1", min_fragment_codons=1
    )

    assert [_tokens(fragment) for fragment in result.fragments] == [
        ["<BOS_CDS>", "ATG", "<EOS_CDS>"],
        ["<BOS_CDS>", "GCT", "GAA", "<EOS_CDS>"],
    ]
    assert [
        (fragment.fragment_index, fragment.codon_start, fragment.codon_end)
        for fragment in result.fragments
    ] == [(0, 0, 1), (1, 2, 4)]
    assert all(fragment.source_id == "gene-1" for fragment in result.fragments)
    assert result.ambiguous_codons == 1
    assert result.source_had_ambiguity is True


@pytest.mark.parametrize(
    ("dna", "expected_ranges", "ambiguous_codons"),
    [
        ("NNNATGAAA", [(1, 3)], 1),
        ("ATGAAANNN", [(0, 2)], 1),
        ("ATGNNNNNNAAA", [(0, 1), (3, 4)], 2),
    ],
)
def test_fragment_tokenizer_handles_boundary_and_consecutive_ambiguity(
    dna, expected_ranges, ambiguous_codons
):
    result = ct.tokenize_cds_fragments(dna, min_fragment_codons=1)

    assert [(f.codon_start, f.codon_end) for f in result.fragments] == expected_ranges
    assert result.ambiguous_codons == ambiguous_codons


def test_fragment_tokenizer_filters_short_fragments_and_counts_partial_tail():
    result = ct.tokenize_cds_fragments(
        "ATGNNNGCTGAANN", min_fragment_codons=2
    )

    assert [(f.fragment_index, f.codon_start, f.codon_end) for f in result.fragments] == [
        (1, 2, 4)
    ]
    assert result.discarded_fragments == 1
    assert result.partial_trailing_bases == 2


def test_tokenizer_cli_writes_fragment_provenance(tmp_path):
    input_path = tmp_path / "cds.txt"
    ids_path = tmp_path / "ids.txt"
    fragments_path = tmp_path / "fragments.tsv"
    input_path.write_text("ATGNNNGCTGAA\nATGAAA\n")

    result = subprocess.run(
        [
            "python",
            "-m",
            "src.codonlm.codon_tokenize",
            "--inp",
            str(input_path),
            "--out_ids",
            str(ids_path),
            "--out_vocab",
            str(tmp_path / "vocab.txt"),
            "--out_itos",
            str(tmp_path / "itos.txt"),
            "--out_fragments",
            str(fragments_path),
            "--min_fragment_codons",
            "1",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert len(ids_path.read_text().splitlines()) == 3
    with open(fragments_path) as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert [(row["source_id"], row["codon_start"], row["codon_end"]) for row in rows] == [
        ("line:0", "0", "1"),
        ("line:0", "2", "4"),
        ("line:1", "0", "2"),
    ]
