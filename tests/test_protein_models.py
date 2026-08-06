import torch
from src.protein_lm.config import ProteinLMConfig, ProteinClassifierConfig
from src.protein_lm.models import ProteinConditionalTransformer, ProteinClassifier

def test_lm_forward_pass():
    """
    Tests that the language model can perform a forward pass and that the output shape is correct.
    """
    config = ProteinLMConfig(vocab_size=30, n_layer=2, n_head=2, n_embd=128, block_size=256, dropout=0.1)
    model = ProteinConditionalTransformer(config)
    model.eval()

    input_ids = torch.randint(0, config.vocab_size, (4, 100))  # batch_size=4, seq_len=100

    with torch.no_grad():
        logits = model(input_ids)

    assert logits.shape == (4, 100, config.vocab_size)

def test_classifier_forward_pass():
    """
    Tests that the classifier can perform a forward pass and that the output shape is correct.
    """
    config = ProteinClassifierConfig(vocab_size=30, n_layer=2, n_head=2, n_embd=128, block_size=256, dropout=0.1, num_classes=2)
    model = ProteinClassifier(config)
    model.eval()

    input_ids = torch.randint(0, config.vocab_size, (4, 100))

    with torch.no_grad():
        logits = model(input_ids)

    assert logits.shape == (4, config.num_classes)


def test_classifier_bos_representation_depends_on_later_residues():
    torch.manual_seed(7)
    config = ProteinClassifierConfig(
        vocab_size=30,
        n_layer=1,
        n_head=2,
        n_embd=16,
        block_size=8,
        dropout=0.0,
        num_classes=2,
    )
    model = ProteinClassifier(config).eval()
    first = torch.tensor([[1, 3, 4, 5, 0, 0]])
    second = torch.tensor([[1, 8, 9, 10, 0, 0]])

    with torch.no_grad():
        first_logits = model(first)
        second_logits = model(second)

    assert not torch.allclose(first_logits, second_logits)


def test_classifier_ignores_explicitly_masked_padding_tokens():
    torch.manual_seed(11)
    config = ProteinClassifierConfig(
        vocab_size=30,
        n_layer=1,
        n_head=2,
        n_embd=16,
        block_size=8,
        dropout=0.0,
        num_classes=2,
    )
    model = ProteinClassifier(config).eval()
    attention_mask = torch.tensor([[1, 1, 1, 0, 0]])
    first = torch.tensor([[1, 3, 4, 0, 0]])
    second = torch.tensor([[1, 3, 4, 8, 9]])

    with torch.no_grad():
        first_logits = model(first, attention_mask)
        second_logits = model(second, attention_mask)

    assert torch.allclose(first_logits, second_logits, atol=1e-6)

def test_causal_mask():
    """
    Tests the causal masking of the language model.
    The logits for a token should only depend on the preceding tokens.
    """
    config = ProteinLMConfig(vocab_size=30, n_layer=2, n_head=2, n_embd=128, block_size=256, dropout=0.1)
    model = ProteinConditionalTransformer(config)
    model.eval() # Disable dropout for deterministic output

    input_ids = torch.randint(0, config.vocab_size, (1, 5))

    # The logits for the first 4 tokens should be identical whether the 5th token is present or not.
    logits_1 = model(input_ids[:, :-1]) # Input of length 4
    logits_2 = model(input_ids)         # Input of length 5

    assert torch.allclose(logits_1, logits_2[:, :-1, :], atol=1e-6)


def test_multitask_classifier_forward_pass():
    from src.protein_lm.models_multi import MultiTaskProteinClassifier
    config = ProteinClassifierConfig(
        vocab_size=30,
        n_layer=2,
        n_head=2,
        n_embd=128,
        block_size=256,
        dropout=0.1,
        num_classes=0,
        pooling="attention",
        bidirectional=True
    )
    task_dims = {"family": 1000, "stability": 2, "function": 500}
    model = MultiTaskProteinClassifier(config, task_dims)
    model.eval()

    input_ids = torch.randint(0, config.vocab_size, (4, 100))
    attention_mask = torch.ones((4, 100), dtype=torch.long)
    attention_mask[:, 80:] = 0  # mock padding

    with torch.no_grad():
        out = model(input_ids, attention_mask=attention_mask)

    assert isinstance(out, dict)
    assert "family" in out
    assert "stability" in out
    assert "function" in out
    assert "attention_weights" in out
    assert out["family"].shape == (4, 1000)
    assert out["stability"].shape == (4, 2)
    assert out["function"].shape == (4, 500)
    assert out["attention_weights"].shape == (4, 100)
