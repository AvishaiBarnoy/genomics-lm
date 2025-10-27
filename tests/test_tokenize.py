from src.codonlm import codon_tokenize as ct


def test_to_ids_basic():
    # ATG (start/M), TTT (F)
    dna = "ATGTTT"
    ids = ct.to_ids(dna)
    # Expect: <bos>, ATG, TTT, <eog>
    assert ids[0] == ct.stoi["<bos>"]
    assert ids[-1] == ct.stoi["<eog>"]
    assert ct.VOCAB[ids[1]] == "ATG"
    assert ct.VOCAB[ids[2]] == "TTT"

