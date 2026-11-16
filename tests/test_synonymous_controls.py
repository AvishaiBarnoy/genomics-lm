"""Unit tests for synonymous controls generation and paired likelihood evaluation."""

import random
import torch
import torch.nn as nn
from src.eval.synonymous_controls import (
    AA_TO_CODONS,
    calculate_token_sequence_nll,
    evaluate_gene_pair_nll,
    reshuffle_synonymous_gene,
    run_synonymous_pilot,
)
from src.eval.translation import CODON_TABLE_11, translate_codons


def test_reshuffle_invariants_uniform_alternative():
    # Construct a sample gene
    codons = ["ATG", "AAA", "GCT", "TTA", "CCT", "GAT", "TGG", "TAA"]
    rng = random.Random(42)

    mutated = reshuffle_synonymous_gene(
        codons,
        rng=rng,
        mode="uniform_alternative",
        preserve_start=True,
        preserve_stop=True,
    )

    # Invariant 1: Length is identical
    assert len(mutated) == len(codons)

    # Invariant 2: Start codon preserved
    assert mutated[0] == codons[0] == "ATG"

    # Invariant 3: Stop codon preserved
    assert mutated[-1] == codons[-1] == "TAA"

    # Invariant 4: Nondegenerate codon (W -> TGG) unchanged
    # TGG is index 6
    assert codons[6] == "TGG"
    assert mutated[6] == "TGG"

    # Invariant 5: Multifold codons changed to alternatives
    # AAA has AAG alternative
    assert mutated[1] == "AAG"

    # Invariant 6: Protein sequence strictly identical
    orig_protein = translate_codons(codons).protein_sequence
    mut_protein = translate_codons(mutated).protein_sequence
    assert orig_protein == mut_protein
    assert len(orig_protein) == 7


def test_reshuffle_string_inputs():
    dna_str = "ATGAAAGCTTTA"
    rng = random.Random(123)
    mutated = reshuffle_synonymous_gene(
        dna_str,
        rng=rng,
        mode="uniform_any",
        preserve_start=True,
        preserve_stop=False,
    )
    assert isinstance(mutated, str)
    assert len(mutated) == len(dna_str)
    assert mutated.startswith("ATG")

    dash_str = "ATG-AAA-GCT-TAA"
    mut_dash = reshuffle_synonymous_gene(dash_str, rng=rng)
    assert isinstance(mut_dash, str)
    assert "-" in mut_dash
    assert mut_dash.startswith("ATG-")
    assert mut_dash.endswith("-TAA")


def test_within_gene_shuffle_preserves_codon_counts():
    # Gene with multiple Leucine and Alanine codons
    codons = [
        "ATG",
        "TTA", "TTG", "CTT", "CTC",  # 4 Leucines
        "GCT", "GCC", "GCT",        # 3 Alanines
        "TAA",
    ]
    rng = random.Random(999)
    mutated = reshuffle_synonymous_gene(
        codons,
        rng=rng,
        mode="within_gene_shuffle",
        preserve_start=True,
        preserve_stop=True,
    )

    assert len(mutated) == len(codons)
    assert mutated[0] == "ATG"
    assert mutated[-1] == "TAA"

    # Check that the counts of each specific codon across the entire gene are identical
    orig_counts = {}
    mut_counts = {}
    for c in codons:
        orig_counts[c] = orig_counts.get(c, 0) + 1
    for c in mutated:
        mut_counts[c] = mut_counts.get(c, 0) + 1

    assert orig_counts == mut_counts

    # Verify identical protein
    assert translate_codons(codons).protein_sequence == translate_codons(mutated).protein_sequence


def test_token_ids_reshuffle():
    # Build mini vocab
    all_codons = sorted(list(CODON_TABLE_11.keys()))
    itos = ["<PAD>", "<BOS_CDS>", "<EOS_CDS>"] + all_codons
    stoi = {tok: i for i, tok in enumerate(itos)}

    gene_codons = ["ATG", "AAA", "GCT", "TAA"]
    gene_ids = [stoi[c] for c in gene_codons]

    rng = random.Random(42)
    mut_ids = reshuffle_synonymous_gene(
        gene_ids,
        rng=rng,
        mode="uniform_alternative",
        preserve_start=True,
        preserve_stop=True,
        itos=itos,
        stoi=stoi,
    )

    assert len(mut_ids) == len(gene_ids)
    assert mut_ids[0] == stoi["ATG"]
    assert mut_ids[-1] == stoi["TAA"]
    mut_codons = [itos[i] for i in mut_ids]
    assert translate_codons(gene_codons).protein_sequence == translate_codons(mut_codons).protein_sequence


def test_short_and_edge_cases():
    # 2-codon gene
    codons = ["ATG", "TAA"]
    mutated = reshuffle_synonymous_gene(codons, rng=1)
    assert mutated == ["ATG", "TAA"]

    # Empty gene
    assert reshuffle_synonymous_gene([]) == []


class DummyModel(nn.Module):
    """Deterministic dummy causal model for testing NLL calculations."""

    def __init__(self, vocab_size: int, favored_token_id: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.favored_token_id = favored_token_id

    def forward(self, x):
        B, T = x.shape
        logits = torch.zeros(B, T, self.vocab_size)
        # Assign high logit to favored token
        logits[:, :, self.favored_token_id] = 10.0
        return logits, None


def test_calculate_token_sequence_nll():
    vocab_size = 10
    model = DummyModel(vocab_size=vocab_size, favored_token_id=5)
    device = torch.device("cpu")

    # Sequence where every target token is 5
    tokens_favored = [1, 5, 5, 5]
    nll_fav, ppl_fav, count_fav = calculate_token_sequence_nll(model, device, tokens_favored)
    assert count_fav == 3
    assert nll_fav < 0.1  # Very low NLL because 5 is favored

    # Sequence where target tokens are NOT 5
    tokens_unfavored = [1, 2, 3, 4]
    nll_unfav, ppl_unfav, count_unfav = calculate_token_sequence_nll(model, device, tokens_unfavored)
    assert count_unfav == 3
    assert nll_unfav > nll_fav


def test_run_synonymous_pilot_mock():
    itos = ["<PAD>", "<BOS_CDS>", "<EOS_CDS>"] + sorted(list(CODON_TABLE_11.keys()))
    stoi = {tok: i for i, tok in enumerate(itos)}

    # Model that prefers AAA over AAG
    model = DummyModel(vocab_size=len(itos), favored_token_id=stoi["AAA"])
    device = torch.device("cpu")

    genes = [
        ("gene_1", ["ATG", "AAA", "AAA", "TAA"]),
        ("gene_2", ["ATG", "AAA", "AAA", "TAA"]),
    ]

    summary = run_synonymous_pilot(
        model,
        device,
        genes,
        itos=itos,
        stoi=stoi,
        mode="uniform_alternative",
        seed=42,
    )

    assert summary.num_genes == 2
    # In uniform_alternative, AAA will be replaced with AAG (unfavored by dummy model)
    # Hence NLL_control > NLL_natural, so delta_nll > 0
    assert summary.median_delta_nll > 0
    assert summary.win_rate == 1.0
    assert summary.success is True
