from __future__ import annotations

import pytest
import torch

from src.codonlm.model_tiny_gpt import CausalSelfAttention


def _attention(*, use_sdpa: bool, dropout: float = 0.4) -> CausalSelfAttention:
    return CausalSelfAttention(
        n_embd=8,
        n_head=2,
        dropout=dropout,
        block_size=6,
        use_sdpa=use_sdpa,
    )


@pytest.mark.parametrize("explicit_mask", [False, True])
def test_sdpa_uses_configured_dropout_only_during_training(
    monkeypatch, explicit_mask
):
    original = torch.nn.functional.scaled_dot_product_attention
    observed = []

    def capture(*args, **kwargs):
        observed.append((kwargs["dropout_p"], kwargs["is_causal"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", capture)
    attention = _attention(use_sdpa=True)
    x = torch.randn(1, 6, 8)
    mask = torch.tril(torch.ones(1, 6, 6, dtype=torch.bool)) if explicit_mask else None

    attention.train()
    attention(x, attn_mask=mask)
    attention.eval()
    attention(x, attn_mask=mask)

    assert observed == [(0.4, not explicit_mask), (0.0, not explicit_mask)]


@pytest.mark.parametrize("explicit_mask", [False, True])
def test_manual_and_sdpa_attention_are_equivalent_in_eval(explicit_mask):
    manual = _attention(use_sdpa=False)
    sdpa = _attention(use_sdpa=True)
    sdpa.load_state_dict(manual.state_dict())
    manual.eval()
    sdpa.eval()
    x = torch.randn(1, 6, 8)
    mask = None
    if explicit_mask:
        mask = torch.tril(torch.ones(1, 6, 6, dtype=torch.bool))
        mask[:, 4:, :3] = False

    manual_output = manual(x, attn_mask=mask)
    sdpa_output = sdpa(x, attn_mask=mask)

    assert torch.allclose(manual_output, sdpa_output, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("use_sdpa", [False, True])
@pytest.mark.parametrize("explicit_mask", [False, True])
def test_attention_dropout_is_seeded_in_train_mode(use_sdpa, explicit_mask):
    attention = _attention(use_sdpa=use_sdpa)
    attention.train()
    x = torch.randn(1, 6, 8)
    mask = torch.tril(torch.ones(1, 6, 6, dtype=torch.bool)) if explicit_mask else None

    torch.manual_seed(123)
    first = attention(x, attn_mask=mask)
    torch.manual_seed(123)
    repeated = attention(x, attn_mask=mask)
    torch.manual_seed(456)
    different = attention(x, attn_mask=mask)

    assert torch.equal(first, repeated)
    assert not torch.equal(first, different)


def test_attention_dropout_adds_no_checkpoint_state():
    no_dropout = _attention(use_sdpa=False, dropout=0.0)
    with_dropout = _attention(use_sdpa=False, dropout=0.4)

    assert no_dropout.state_dict().keys() == with_dropout.state_dict().keys()
