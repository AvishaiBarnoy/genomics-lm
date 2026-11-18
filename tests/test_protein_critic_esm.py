"""Unit tests for ESM-2 (8M) external protein critic."""

import math
import pytest
from src.eval.protein_critic_esm import (
    ESM2ProteinCritic,
    run_esm2_critic_benchmark,
    translate_and_score_cds,
)


@pytest.fixture(scope="module")
def critic():
    """Shared ESM-2 critic instance on CPU for testing."""
    return ESM2ProteinCritic(device="cpu")


def test_esm2_critic_score_finiteness(critic):
    # Test peptide (7 amino acids)
    seq = "MKALVAG"
    score, k = critic.score_protein(seq, num_positions=5, seed=42)
    assert k == 5
    assert math.isfinite(score)
    # Log-likelihood of masked tokens is always non-positive (in nats)
    assert score <= 0.0
    assert score > -20.0


def test_esm2_critic_deterministic_seed(critic):
    seq = "MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGEEHFKGLVLIAFSQYLQQCPFDEHVKLVNELTEFAKTCV"
    score1, k1 = critic.score_protein(seq, num_positions=16, seed=123)
    score2, k2 = critic.score_protein(seq, num_positions=16, seed=123)
    assert k1 == 16
    assert k2 == 16
    assert score1 == score2

    # Different seed evaluates different positions
    score3, _ = critic.score_protein(seq, num_positions=16, seed=999)
    assert score1 != score3


def test_short_protein_handling(critic):
    # 4 residues: should evaluate exactly 4 positions when num_positions=16
    seq = "MKAW"
    score, k = critic.score_protein(seq, num_positions=16, seed=42)
    assert k == 4
    assert math.isfinite(score)


def test_empty_protein_handling(critic):
    score, k = critic.score_protein("", num_positions=16, seed=42)
    assert k == 0
    assert score == float("-inf")


def test_translate_and_score_cds(critic):
    cds = "ATG-AAA-GCT-CTG-GTA-GCA-GGT-TAA"
    score, report = translate_and_score_cds(cds, critic=critic, num_positions=5, seed=42)
    assert report.is_valid_orf is True
    assert report.protein_sequence == "MKALVAG"
    assert math.isfinite(score)


def test_benchmark_summary(critic):
    # Real natural peptide vs random degraded peptide
    real_peptides = [("real_1", "MKALVAGMKAW"), ("real_2", "MKLVAGMKALW")]
    gen_peptides = [("gen_1", "WWWWWWWWWWW"), ("gen_2", "CCCCCCCCCCC")]

    summary = run_esm2_critic_benchmark(
        critic,
        gen_peptides,
        real_peptides,
        num_positions=8,
        seed=42,
    )

    assert summary.num_generated == 2
    assert summary.num_real == 2
    assert summary.all_scores_finite is True
    assert summary.delta_median_score == summary.median_real_score - summary.median_generated_score
    assert len(summary.results) == 4
