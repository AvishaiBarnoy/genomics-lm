import pytest
import torch

from src.codonlm.model_tiny_gpt import TinyGPT


@pytest.mark.parametrize("tie", [True, False])
def test_tie_embeddings_toggle(tie):
    m = TinyGPT(vocab_size=32, block_size=16, n_layer=1, n_head=2, n_embd=16, dropout=0.0, tie_embeddings=tie)
    x = torch.randint(0, 32, (2, 8))
    logits, loss = m(x, x)
    assert logits.shape == (2, 8, 32)


def test_gqa_forward():
    # n_kv_head divides n_head
    m = TinyGPT(vocab_size=32, block_size=16, n_layer=1, n_head=4, n_embd=32, dropout=0.0, n_kv_head=2)
    x = torch.randint(0, 32, (2, 8))
    logits, _ = m(x)
    assert logits.shape == (2, 8, 32)


@pytest.mark.xfail(strict=False, reason="SDPA may be unavailable on this build")
def test_sdpa_forward():
    m = TinyGPT(vocab_size=32, block_size=16, n_layer=1, n_head=2, n_embd=16, dropout=0.0, use_sdpa=True)
    x = torch.randint(0, 32, (2, 8))
    logits, _ = m(x)
    assert logits.shape == (2, 8, 32)

