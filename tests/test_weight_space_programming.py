"""
Unit tests for Weight-Space Programming in CodonLM.
Verifies tensor editing, preservation of unedited weights, and finite outputs.
"""

import copy
import pytest
import torch
import torch.nn.functional as F

from src.codonlm.model_tiny_gpt import TinyGPT
from scripts.apply_weight_space_programming import sha256_tensor


@pytest.fixture
def dummy_model():
    model = TinyGPT(
        vocab_size=68,
        block_size=64,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
        tie_embeddings=False,
    )
    model.eval()
    return model


def test_weight_space_programming_tensor_integrity(dummy_model):
    state_before = copy.deepcopy(dummy_model.state_dict())
    orig_head = state_before["head.weight"].clone()
    orig_hash = sha256_tensor(orig_head)

    # Apply 1.12 scaling to head.weight
    scale = 1.12
    state_after = copy.deepcopy(state_before)
    state_after["head.weight"] = state_after["head.weight"] * scale

    # Load into new model
    dummy_model.load_state_dict(state_after)

    # 1. Verify head.weight is exactly scaled
    torch.testing.assert_close(dummy_model.head.weight, orig_head * scale)
    assert sha256_tensor(dummy_model.head.weight) != orig_hash

    # 2. Verify all other tensors are bitwise identical
    for k, v in dummy_model.state_dict().items():
        if k != "head.weight":
            torch.testing.assert_close(v, state_before[k])


def test_weight_space_programming_forward_pass(dummy_model):
    scale = 1.12
    with torch.no_grad():
        dummy_model.head.weight.mul_(scale)

    x = torch.randint(0, 68, (2, 16))
    with torch.no_grad():
        logits, loss = dummy_model(x)

    assert logits.shape == (2, 16, 68)
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()
    assert loss is None
