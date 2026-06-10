from __future__ import annotations
import pytest
import torch
from src.eval.inference_playground import (
    query_next_codon, 
    generate_cds, 
    classify_protein, 
    translate_codons_to_aa, 
    score_protein_sequence, 
    PROTEIN_AVAILABLE
)
from src.codonlm.model_tiny_gpt import TinyGPT

if PROTEIN_AVAILABLE:
    from src.protein_lm.tokenizer import ProteinTokenizer
    from src.protein_lm.models_multi import MultiTaskProteinClassifier
    from src.protein_lm.config import ProteinClassifierConfig
    from src.protein_lm.models import ProteinConditionalTransformer
    from src.protein_lm.config import ProteinLMConfig


def test_query_next_codon():
    """Tests that query_next_codon correctly predicts next token probabilities using a small model."""
    vocab_size = 69
    block_size = 16
    model = TinyGPT(vocab_size=vocab_size, block_size=block_size, n_layer=1, n_head=1, n_embd=16)
    model.eval()
    
    itos = [f"tok_{i}" for i in range(vocab_size)]
    itos[1] = "<BOS_CDS>"
    stoi = {tok: i for i, tok in enumerate(itos)}
    
    device = torch.device("cpu")
    res = query_next_codon(model, stoi, itos, device, "tok_4 tok_5", top_k=3)
    
    assert len(res) == 3
    assert "token" in res[0]
    assert "prob" in res[0]
    assert isinstance(res[0]["prob"], float)


def test_generate_cds():
    """Tests sequence generation using a dummy TinyGPT model."""
    vocab_size = 69
    block_size = 16
    model = TinyGPT(vocab_size=vocab_size, block_size=block_size, n_layer=1, n_head=1, n_embd=16)
    model.eval()
    
    itos = [f"tok_{i}" for i in range(vocab_size)]
    itos[1] = "<BOS_CDS>"
    itos[2] = "<EOS_CDS>"
    # Add a stop codon
    itos[5] = "TAA"
    stoi = {tok: i for i, tok in enumerate(itos)}
    
    device = torch.device("cpu")
    tokens, info = generate_cds(
        model, stoi, itos, device, 
        dna_prefix="tok_4", 
        max_new_tokens=5, 
        temperature=1.0, 
        top_k=0, 
        stop_on_eos=True, 
        stop_on_bio_stop=True
    )
    
    assert len(tokens) > 1
    assert isinstance(info, dict)
    assert "hit_eos" in info
    assert "hit_bio_stop" in info


def test_translate_codons_to_aa():
    """Tests translation of codons to amino acids, stopping at stop codons."""
    codons = ["ATG", "GCT", "AAC", "TAA", "GCT"]
    aa = translate_codons_to_aa(codons)
    assert aa == "MAN"


@pytest.mark.skipif(not PROTEIN_AVAILABLE, reason="protein_lm package dependencies not available")
def test_classify_protein():
    """Tests the protein classifier helper using a small MultiTaskProteinClassifier."""
    tokenizer = ProteinTokenizer()
    task_dims = {
        "family": 5,
        "function": 4,
        "stability": 3
    }
    config = ProteinClassifierConfig(
        vocab_size=len(tokenizer.vocab),
        block_size=64,
        n_layer=1,
        n_head=1,
        n_embd=16,
        dropout=0.0,
        num_classes=0
    )
    model = MultiTaskProteinClassifier(config, task_dims)
    model.eval()
    
    itos_dict = {
        "family": {i: f"fam_{i}" for i in range(5)},
        "function": {i: f"func_{i}" for i in range(4)},
        "stability": {i: f"stab_{i}" for i in range(3)}
    }
    
    device = torch.device("cpu")
    results = classify_protein(model, tokenizer, itos_dict, device, "MQA")
    
    assert "family" in results
    assert "function" in results
    assert "stability" in results
    assert "prediction" in results["family"]
    assert "probability" in results["family"]


@pytest.mark.skipif(not PROTEIN_AVAILABLE, reason="protein_lm package dependencies not available")
def test_score_protein_sequence():
    """Tests scoring a protein sequence with a dummy ProteinConditionalTransformer."""
    tokenizer = ProteinTokenizer()
    config = ProteinLMConfig(
        vocab_size=len(tokenizer.vocab),
        n_layer=1,
        n_head=1,
        n_embd=16,
        block_size=64,
        dropout=0.0
    )
    model = ProteinConditionalTransformer(config)
    model.eval()
    
    device = torch.device("cpu")
    scores = score_protein_sequence(model, tokenizer, device, "MAN")
    assert "perplexity" in scores
    assert "avg_log_prob" in scores
    assert isinstance(scores["perplexity"], float)


def test_get_attention_weights():
    """Tests attention weight extraction using a dummy TinyGPT model."""
    from src.eval.inference_playground import get_attention_weights
    vocab_size = 69
    block_size = 16
    model = TinyGPT(vocab_size=vocab_size, block_size=block_size, n_layer=1, n_head=1, n_embd=16)
    model.eval()
    
    itos = [f"TOK_{i}" for i in range(vocab_size)]
    itos[1] = "<BOS_CDS>"
    stoi = {tok: i for i, tok in enumerate(itos)}
    
    device = torch.device("cpu")
    res = get_attention_weights(model, stoi, itos, device, "tok_4 tok_5")
    
    assert res is not None
    assert "tokens" in res
    assert "attention" in res
    assert "Layer 1" in res["attention"]
    assert res["attention"]["Layer 1"].shape == (1, 3, 3)  # n_head=1, T=3 (BOS + TOK_4 + TOK_5)


