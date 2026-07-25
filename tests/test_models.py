import torch
import pytest

from src.codonlm.model_tiny_gpt import TinyGPT


def test_tinygpt_forward_shapes():
    vocab, T, B = 69, 16, 4
    m = TinyGPT(vocab_size=vocab, block_size=T, n_layer=1, n_head=1, n_embd=32, dropout=0.0, use_checkpoint=False)
    x = torch.randint(0, vocab, (B, T), dtype=torch.long)
    y = x.clone()
    y[:, 0] = 0  # pad to exercise ignore_index=0 in loss
    logits, loss = m(x, y)
    assert logits.shape == (B, T, vocab)
    assert loss is not None and torch.isfinite(loss)


def test_tinygpt_sdpa():
    vocab, T, B = 69, 16, 4
    m = TinyGPT(vocab_size=vocab, block_size=T, n_layer=1, n_head=1, n_embd=32, dropout=0.0, use_checkpoint=False, use_sdpa=True)
    x = torch.randint(0, vocab, (B, T), dtype=torch.long)
    y = x.clone()
    y[:, 0] = 0  # pad to exercise ignore_index=0 in loss
    logits, loss = m(x, y)
    assert logits.shape == (B, T, vocab)
    assert loss is not None and torch.isfinite(loss)


def test_attention_window_and_separator_mask_match_context_contract():
    model = TinyGPT(
        vocab_size=8,
        block_size=5,
        n_layer=1,
        n_head=1,
        n_embd=8,
        dropout=0.0,
        sep_id=3,
    )
    tokens = torch.tensor([[1, 4, 3, 5, 6]])

    full = model.build_attention_mask(tokens)[0, 0]
    assert full[1, 0]
    assert not full[3, 1]
    assert full[3, 2]
    assert full[4, 2]

    local = model.build_attention_mask(tokens, attention_window=1)[0, 0]
    assert torch.equal(local, torch.eye(5, dtype=torch.bool))

    with pytest.raises(ValueError, match="at least 1"):
        model.build_attention_mask(tokens, attention_window=0)


def test_tinygpt_swiglu_shapes():
    vocab, T, B = 69, 16, 4
    m = TinyGPT(
        vocab_size=vocab,
        block_size=T,
        n_layer=1,
        n_head=1,
        n_embd=32,
        dropout=0.0,
        use_checkpoint=False,
        use_swiglu=True
    )
    x = torch.randint(0, vocab, (B, T), dtype=torch.long)
    logits, loss = m(x, x)
    assert logits.shape == (B, T, vocab)
    assert loss is not None and torch.isfinite(loss)


def test_tinygpt_rope_shapes():
    vocab, T, B = 69, 16, 4
    m = TinyGPT(
        vocab_size=vocab,
        block_size=T,
        n_layer=1,
        n_head=1,
        n_embd=32,
        dropout=0.0,
        use_checkpoint=False,
        use_rope=True
    )
    x = torch.randint(0, vocab, (B, T), dtype=torch.long)
    logits, loss = m(x, x)
    assert logits.shape == (B, T, vocab)
    assert loss is not None and torch.isfinite(loss)
