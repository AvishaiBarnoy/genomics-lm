import pytest
from src.eval.translation import (
    translate_codons,
    batch_translate,
    CODON_TABLE_11,
    STOP_CODONS,
    SPECIAL_TOKENS,
)


def test_valid_coding_sequence():
    """A valid CDS with start codon ATG and stop codon TAA translates cleanly."""
    # ATG -> M, AAA -> K, GCT -> A, TAA -> *
    seq = "<BOS_CDS>-ATG-AAA-GCT-TAA-<EOS_CDS>"
    report = translate_codons(seq)

    assert report.num_codons == 4
    assert report.num_invalid_codons == 0
    assert report.invalid_codons == []
    assert report.protein_sequence == "MKA"
    assert report.translated_length == 3
    assert report.starts_with_met is True
    assert report.has_terminal_stop is True
    assert report.internal_stop_count == 0
    assert report.internal_stop_positions == []
    assert report.is_valid_orf is True


def test_internal_stop_codon():
    """A sequence with an internal stop codon (e.g. TAG midway) is correctly flagged."""
    # ATG -> M, TAG -> * (internal), GCT -> A, TAA -> * (terminal)
    seq = "ATG-TAG-GCT-TAA"
    report = translate_codons(seq)

    assert report.num_codons == 4
    assert report.num_invalid_codons == 0
    assert report.protein_sequence == "MA"
    assert report.translated_length == 2
    assert report.starts_with_met is True
    assert report.has_terminal_stop is True
    assert report.internal_stop_count == 1
    assert report.internal_stop_positions == [1]
    assert report.is_valid_orf is False


def test_invalid_codons():
    """Codons with invalid bases or wrong lengths are recorded as invalid."""
    # ATG -> M, XYZ -> invalid, NNN -> invalid, GCT -> A, TAA -> *
    seq = "ATG XYZ NNN GCT TAA"
    report = translate_codons(seq)

    assert report.num_codons == 5
    assert report.num_invalid_codons == 2
    assert report.invalid_codons == ["XYZ", "NNN"]
    assert report.protein_sequence == "MA"
    assert report.translated_length == 2
    assert report.is_valid_orf is False


def test_special_token_stripping():
    """Tokenizer special tokens (<BOS_CDS>, <EOS_CDS>, <PAD>, <SEP>, <UNK>) are removed."""
    tokens = ["<PAD>", "<BOS_CDS>", "ATG", "<SEP>", "AAA", "<UNK>", "TAA", "<EOS_CDS>"]
    report = translate_codons(tokens)

    assert report.raw_tokens == tokens
    assert report.num_codons == 3
    assert report.protein_sequence == "MK"
    assert report.has_terminal_stop is True
    assert report.is_valid_orf is True


def test_empty_sequence():
    """Empty or special-token-only input returns empty report."""
    report = translate_codons("<BOS_CDS> <EOS_CDS>")
    assert report.num_codons == 0
    assert report.translated_length == 0
    assert report.protein_sequence == ""
    assert report.is_valid_orf is False


def test_multiple_internal_stops():
    """Multiple internal stops are counted and their positions logged."""
    seq = "ATG-TGA-GCT-TAG-TAA"
    # index 0: ATG (M)
    # index 1: TGA (*) -> internal
    # index 2: GCT (A)
    # index 3: TAG (*) -> internal
    # index 4: TAA (*) -> terminal
    report = translate_codons(seq)
    assert report.internal_stop_count == 2
    assert report.internal_stop_positions == [1, 3]
    assert report.has_terminal_stop is True


def test_batch_translate():
    """Batch translation returns a report per sequence with proper sequence indices."""
    seqs = ["ATG-AAA-TAA", "ATG-XYZ-TAA"]
    reports = batch_translate(seqs)

    assert len(reports) == 2
    assert reports[0].sequence_index == 0
    assert reports[0].is_valid_orf is True
    assert reports[1].sequence_index == 1
    assert reports[1].num_invalid_codons == 1

    d0 = reports[0].to_dict()
    assert "sequence_index" in d0
    assert d0["protein_sequence"] == "MK"
