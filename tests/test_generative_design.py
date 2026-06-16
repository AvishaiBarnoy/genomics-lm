"""
tests/test_generative_design.py

Unit tests for module-level functions in scripts/generative_design_loop.py.
All tests are pure-Python / pure-PyTorch — no checkpoints or external models required.
"""

import math
import unittest
from unittest.mock import MagicMock, patch

import torch

from scripts.generative_design_loop import (
    gc_content,
    kmer_diversity,
    pairwise_identity,
    red_generate,
    translate_dna,
)


# ─────────────────────────────────────────────────────────────────────────────
# translate_dna
# ─────────────────────────────────────────────────────────────────────────────

class TestTranslateDNA(unittest.TestCase):

    def test_translate_dna_basic(self):
        """ATG (M) + some coding codons + TAA stop → correct AA string + terminated=True."""
        # ATG=M, AAG=K, GTC=V, TAA=stop
        dna = "ATGAAGGTATAA"
        aa, terminated = translate_dna(dna)
        self.assertEqual(aa, "MKV")
        self.assertTrue(terminated)

    def test_translate_dna_no_stop(self):
        """Sequence without a stop codon → terminated=False, all AAs returned."""
        # ATG=M, AAG=K, GTA=V  (no stop codon)
        dna = "ATGAAGGTA"
        aa, terminated = translate_dna(dna)
        self.assertEqual(aa, "MKV")
        self.assertFalse(terminated)

    def test_translate_dna_unknown_codon(self):
        """A codon not in the table (e.g. NNN) → 'X' amino acid."""
        dna = "ATGNNN"
        aa, terminated = translate_dna(dna)
        self.assertIn("M", aa)
        self.assertIn("X", aa)
        self.assertFalse(terminated)


# ─────────────────────────────────────────────────────────────────────────────
# pairwise_identity
# ─────────────────────────────────────────────────────────────────────────────

class TestPairwiseIdentity(unittest.TestCase):

    def test_pairwise_identity_identical(self):
        """Two identical sequences → identity = 1.0."""
        seqs = ["MKVLST", "MKVLST"]
        result = pairwise_identity(seqs)
        self.assertAlmostEqual(result, 1.0, places=5)

    def test_pairwise_identity_different(self):
        """Two completely different sequences → identity near 0.0."""
        # 'M' vs 'K', 'K' vs 'V', ... — use completely mismatched chars
        seqs = ["MKVLST", "YWRPAI"]
        result = pairwise_identity(seqs)
        self.assertAlmostEqual(result, 0.0, places=5)


# ─────────────────────────────────────────────────────────────────────────────
# kmer_diversity
# ─────────────────────────────────────────────────────────────────────────────

class TestKmerDiversity(unittest.TestCase):

    def test_kmer_diversity_single(self):
        """A single sequence with exactly one unique 3-mer → 1/8000 coverage."""
        # "MKV" has exactly one 3-mer: "MKV"
        seqs = ["MKV"]
        result = kmer_diversity(seqs, k=3)
        expected = 1 / (20 ** 3)   # = 1/8000
        self.assertAlmostEqual(result, expected, places=8)


# ─────────────────────────────────────────────────────────────────────────────
# gc_content
# ─────────────────────────────────────────────────────────────────────────────

class TestGCContent(unittest.TestCase):

    def test_gc_content(self):
        """ATG = 1 G out of 3 bases → GC = 1/3; GGG = 3 G → GC = 1.0."""
        codon_seqs = [["ATG"], ["GGG"]]
        results = gc_content(codon_seqs)
        self.assertEqual(len(results), 2)
        self.assertAlmostEqual(results[0], 1 / 3, places=5)   # ATG: 1 G, 0 C
        self.assertAlmostEqual(results[1], 1.0, places=5)      # GGG: 3 G


# ─────────────────────────────────────────────────────────────────────────────
# red_generate (mocked)
# ─────────────────────────────────────────────────────────────────────────────

class TestRedGenerateMocked(unittest.TestCase):

    def test_red_generate_mocked(self):
        """Mock the model to always emit a biological stop codon on the first step.
        Expect terminated=True and n_attempts=1.
        """
        device = torch.device("cpu")

        # Vocab: index 0=<PAD>, 1=<BOS_CDS>, 2=<EOS_CDS>, 3=ATG, 4=TAA (stop)
        itos = ["<PAD>", "<BOS_CDS>", "<EOS_CDS>", "ATG", "TAA"]
        stoi = {tok: i for i, tok in enumerate(itos)}

        vocab_size = len(itos)

        # Build a logit vector that puts all probability mass on TAA (index 4)
        def fake_forward(x):
            batch, seq_len = x.shape
            logits = torch.full((batch, seq_len, vocab_size), float("-inf"))
            logits[:, :, 4] = 0.0   # index 4 = "TAA" stop codon
            return logits, None

        mock_model = MagicMock()
        mock_model.side_effect = fake_forward

        codons, terminated, n_attempts = red_generate(
            model=mock_model,
            device=device,
            stoi=stoi,
            itos=itos,
            max_codons=50,
            max_attempts=5,
            temperature=1.0,
            top_k=0,      # disable top-k so full distribution is used
            min_aa_length=0,  # no minimum — test pure termination logic
        )

        self.assertTrue(terminated, "Expected terminated=True when stop codon always emitted")
        self.assertEqual(n_attempts, 1, "Expected only 1 attempt needed")
        # No coding codons — sequence starts with stop immediately
        self.assertIsInstance(codons, list)


if __name__ == "__main__":
    unittest.main()
