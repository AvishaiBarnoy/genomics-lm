from __future__ import annotations
import pytest
from src.codonlm.hybrid_tokenizer import HybridTokenizer

def test_vocab_initialization():
    tokenizer = HybridTokenizer()
    assert tokenizer.vocab_size == 74
    
    # Check special tokens
    assert tokenizer.stoi["<PAD>"] == 0
    assert tokenizer.stoi["<BOS_CDS>"] == 1
    assert tokenizer.stoi["<EOS_CDS>"] == 2
    assert tokenizer.stoi["<UNK>"] == 3
    assert tokenizer.stoi["<UTR_START>"] == 4
    assert tokenizer.stoi["<UTR_END>"] == 5
    
    # Check codon bounds
    assert "ATG" in tokenizer.stoi
    assert tokenizer.stoi["ATG"] >= 6
    assert tokenizer.stoi["ATG"] < 70
    
    # Check nucleotides
    assert "A" in tokenizer.stoi
    assert tokenizer.stoi["A"] >= 70
    assert tokenizer.stoi["A"] < 74

def test_encode_decode_basic():
    tokenizer = HybridTokenizer()
    dna = "ATGGCCTAA"
    intervals = [(0, 9, "+")]
    
    encoded = tokenizer.encode(dna, intervals)
    
    # Expected: <BOS_CDS>, ATG, GCC, TAA, <EOS_CDS>
    expected_tokens = [
        "<BOS_CDS>",
        "ATG",
        "GCC",
        "TAA",
        "<EOS_CDS>"
    ]
    expected_ids = [tokenizer.stoi[tok] for tok in expected_tokens]
    
    assert encoded == expected_ids
    
    decoded = tokenizer.decode(encoded)
    assert decoded == dna
    
    decoded_genomic = tokenizer.decode_genomic(encoded, intervals)
    assert decoded_genomic == dna

def test_encode_decode_with_utr():
    tokenizer = HybridTokenizer()
    # 4 bp UTR + 6 bp CDS + 6 bp UTR = 16 bp total
    dna = "ACGTATGGCCAGCTGA"
    intervals = [(4, 10, "+")]
    
    encoded = tokenizer.encode(dna, intervals)
    
    # Expected:
    # <UTR_START>, A, C, G, T, <UTR_END>,
    # <BOS_CDS>, ATG, GCC, <EOS_CDS>,
    # <UTR_START>, A, G, C, T, G, A, <UTR_END>
    expected_tokens = [
        "<UTR_START>", "A", "C", "G", "T", "<UTR_END>",
        "<BOS_CDS>", "ATG", "GCC", "<EOS_CDS>",
        "<UTR_START>", "A", "G", "C", "T", "G", "A", "<UTR_END>"
    ]
    expected_ids = [tokenizer.stoi[tok] for tok in expected_tokens]
    
    assert encoded == expected_ids
    
    # Simple decode gives coding-oriented sequence (which matches in forward strand)
    assert tokenizer.decode(encoded) == dna
    
    # Genomic decode gives original DNA
    assert tokenizer.decode_genomic(encoded, intervals) == dna

def test_reverse_strand():
    tokenizer = HybridTokenizer()
    # 3 bp UTR + 6 bp CDS + 2 bp UTR = 11 bp total
    # Genomic CDS: GGCTAC -> RC is GTAGCC (GTA, GCC)
    dna = "ATCGGCTACCC"
    intervals = [(3, 9, "-")]
    
    encoded = tokenizer.encode(dna, intervals)
    
    # Expected:
    # <UTR_START>, A, T, C, <UTR_END>,
    # <BOS_CDS>, GTA, GCC, <EOS_CDS>,
    # <UTR_START>, C, C, <UTR_END>
    expected_tokens = [
        "<UTR_START>", "A", "T", "C", "<UTR_END>",
        "<BOS_CDS>", "GTA", "GCC", "<EOS_CDS>",
        "<UTR_START>", "C", "C", "<UTR_END>"
    ]
    expected_ids = [tokenizer.stoi[tok] for tok in expected_tokens]
    
    assert encoded == expected_ids
    
    # Simple decode returns the RC coding sequence: ATC + GTAGCC + CC
    assert tokenizer.decode(encoded) == "ATCGTAGCCCC"
    
    # Genomic decode restores the original genomic sequence: ATCGGCTACCC
    assert tokenizer.decode_genomic(encoded, intervals) == dna

def test_overlap_error():
    tokenizer = HybridTokenizer()
    dna = "ATGGCCTAA"
    intervals = [(0, 6, "+"), (3, 9, "+")]  # overlapping
    
    with pytest.raises(ValueError, match="Overlapping CDS intervals are not supported"):
        tokenizer.encode(dna, intervals)
