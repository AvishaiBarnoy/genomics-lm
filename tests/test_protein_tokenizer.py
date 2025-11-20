from src.protein_lm.tokenizer import ProteinTokenizer

def test_round_trip_sequence():
    """
    Tests that encoding and then decoding a sequence returns the original sequence.
    """
    tokenizer = ProteinTokenizer()
    sequence = "ARNDCQEGHILKMFPSTWYV"
    encoded = tokenizer.encode_sequence(sequence)
    decoded = tokenizer.decode_sequence(encoded)
    assert sequence == decoded

def test_unknown_amino_acid():
    """
    Tests that the tokenizer correctly handles unknown amino acids.
    """
    tokenizer = ProteinTokenizer()
    sequence = "ARNDX"
    encoded = tokenizer.encode_sequence(sequence)
    assert encoded == [
        tokenizer.token_to_id['A'],
        tokenizer.token_to_id['R'],
        tokenizer.token_to_id['N'],
        tokenizer.token_to_id['D'],
        tokenizer.token_to_id['X']
    ]
    decoded = tokenizer.decode_sequence(encoded)
    assert "ARNDX" == decoded

def test_encode_conditions():
    """
    Tests that condition tokens are correctly encoded.
    """
    tokenizer = ProteinTokenizer()
    conditions = ["<FUNC:ENZYME>", "<TOPO:TM>"]
    encoded = tokenizer.encode_conditions(conditions)
    assert encoded == [
        tokenizer.token_to_id["<FUNC:ENZYME>"],
        tokenizer.token_to_id["<TOPO:TM>"]
    ]

def test_special_tokens():
    """
    Tests that the special token IDs can be accessed.
    """
    tokenizer = ProteinTokenizer()
    assert tokenizer.bos_token_id is not None
    assert tokenizer.eos_token_id is not None
    assert tokenizer.pad_token_id is not None

def test_vocab_size():
    """
    Tests that the vocabulary size is correct.
    """
    tokenizer = ProteinTokenizer()
    # 20 AAs + 1 unknown + 3 special + 4 conditions
    assert len(tokenizer) == 28