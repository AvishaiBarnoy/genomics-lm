from src.codonlm.codon_tokenize import VOCAB, stoi, itos, to_ids, STOP_CODONS
import torch


def test_vocab_specials_once():
    specials = ["<PAD>", "<BOS_CDS>", "<EOS_CDS>", "<SEP>"]
    for s in specials:
        assert list(VOCAB).count(s) == 1


def test_to_ids_roundtrip_minimal():
    seq = "ATGAAATGA"  # ATG AAA TGA
    ids = to_ids(seq)
    toks = [itos[i] for i in ids]
    assert toks[0] == "<BOS_CDS>"
    assert toks[-1] == "<EOS_CDS>"
    assert "ATG" in toks and "TGA" in toks
    assert STOP_CODONS == {"TAA","TAG","TGA"}


def test_segment_mask_blocking():
    # Simulate indices with a <SEP> boundary and verify block across segments
    sep_id = stoi["<SEP>"]
    bos = stoi["<BOS_CDS>"]
    a = stoi["ATG"]; b = stoi["GCT"]
    # [BOS a b SEP BOS a]
    idx = torch.tensor([[bos, a, b, sep_id, bos, a]])
    sep = (idx == sep_id)
    seg = torch.cumsum(sep, dim=1)
    mask = (seg.unsqueeze(-1) == seg.unsqueeze(-2))  # (B,T,T)
    # positions before and after SEP must be masked out (False equality), diagonals within each side True
    assert mask[0, 1, 2]  # within same left segment
    assert not mask[0, 1, 4]  # across SEP
    assert mask[0, 4, 5]  # within right segment

