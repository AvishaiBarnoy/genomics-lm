import torch
import pytest
from src.codonlm.generate import generate_cds_synonymous, AA_TO_CODONS, CODON_TABLE


class DummyCausalModel(torch.nn.Module):
    """Dummy model that outputs uniform logits for testing synonymous masking."""
    def __init__(self, vocab_size: int, block_size: int = 512):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size

    def forward(self, x, y=None, return_aux: bool = False):
        T = x.shape[1]
        logits = torch.zeros((1, T, self.vocab_size), dtype=torch.float32, device=x.device)
        if return_aux:
            return logits, None, {}
        return logits, None


def test_aa_to_codons_mapping():
    """Verify that the amino-acid-to-codon reverse map is correctly populated."""
    assert "M" in AA_TO_CODONS
    assert "ATG" in AA_TO_CODONS["M"]
    assert len(AA_TO_CODONS["M"]) == 1

    assert "L" in AA_TO_CODONS
    assert "TTA" in AA_TO_CODONS["L"]
    assert "TTG" in AA_TO_CODONS["L"]
    assert "CTT" in AA_TO_CODONS["L"]

    assert "_" in AA_TO_CODONS
    assert "TAA" in AA_TO_CODONS["_"]


def test_synonymous_constrained_generation():
    """Verify end-to-end synonymous constrained codon generation."""
    # Build a standard hybrid vocabulary containing codons, stop codons, and EOS
    itos = ["<PAD>", "<BOS_CDS>", "<EOS_CDS>", "<SEP>"]
    # Add all codons from table
    for codon in sorted(CODON_TABLE.keys()):
        itos.append(codon)
    stoi = {t: i for i, t in enumerate(itos)}

    model = DummyCausalModel(vocab_size=len(itos))
    device = torch.device("cpu")

    # Prefix starting with BOS
    ctx_ids = [stoi["<BOS_CDS>"]]

    # Generate for target protein "MKAQ"
    target_protein = "MKAQ"
    gen_ids, info = generate_cds_synonymous(
        model=model,
        critic_model=None,
        c_tokenizer=None,
        device=device,
        ctx_ids=ctx_ids,
        stoi=stoi,
        itos=itos,
        target_protein=target_protein,
        alpha=0.0,
    )

    # 1. Verify correct token lengths:
    # ids should contain: BOS + 4 (residues) + 1 (stop) + 1 (EOS) = 7 tokens
    assert len(gen_ids) == 7
    assert itos[gen_ids[0]] == "<BOS_CDS>"
    assert itos[gen_ids[-1]] == "<EOS_CDS>"

    # 2. Verify translation correctness:
    # Codons generated (tokens 1 to 4 should map to residues M, K, A, Q)
    res_m = CODON_TABLE[itos[gen_ids[1]]]
    res_k = CODON_TABLE[itos[gen_ids[2]]]
    res_a = CODON_TABLE[itos[gen_ids[3]]]
    res_q = CODON_TABLE[itos[gen_ids[4]]]

    assert res_m == "M"
    assert res_k == "K"
    assert res_a == "A"
    assert res_q == "Q"

    # Token 5 must be a stop codon
    res_stop = CODON_TABLE[itos[gen_ids[5]]]
    assert res_stop == "_"

    # 3. Verify returned info:
    assert info["generated_codons"] == 5
    assert info["target_codons"] == 5
    assert info["had_terminal_stop"] is True
