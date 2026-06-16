import torch
import torch.nn.functional as F
import pytest
from src.codonlm.model_tiny_gpt import NoPropTinyGPT, NoPropBlock

def test_noprop_initialization():
    vocab_size = 64
    block_size = 32
    n_layer = 3
    n_head = 2
    n_embd = 32

    model = NoPropTinyGPT(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        sep_id=3
    )

    assert len(model.blocks) == n_layer
    assert isinstance(model.blocks[0], NoPropBlock)
    assert model.tok_emb.weight.shape == (vocab_size, n_embd)

def test_noprop_gradient_isolation():
    vocab_size = 64
    block_size = 10
    n_layer = 3
    n_head = 2
    n_embd = 32

    model = NoPropTinyGPT(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd
    )

    # Mock inputs and targets
    idx = torch.randint(1, vocab_size, (2, block_size))
    targets = torch.randint(1, vocab_size, (2, block_size))

    # Forward embeddings
    pos = torch.arange(0, block_size).unsqueeze(0)
    h = model.tok_emb(idx) + model.pos_emb(pos)

    y_clean = model.tok_emb(targets)
    noise = torch.randn_like(y_clean) * 0.1
    y_noisy = y_clean + noise

    # Check block 1 backprop isolation
    # Forward block 0
    h0_out, pred_y0 = model.blocks[0](h, noisy_targets=y_noisy)

    # Forward block 1 (with detached input from block 0)
    h1_in = h0_out.detach()
    h1_out, pred_y1 = model.blocks[1](h1_in, noisy_targets=y_noisy)

    # Compute local MSE loss for block 1
    loss_b1 = F.mse_loss(pred_y1, y_clean)
    loss_b1.backward()

    # Assertions:
    # 1. Block 1 parameters must have gradients.
    # 2. Block 0 and Block 2 parameters must NOT have gradients (grad is None).
    for name, param in model.blocks[1].named_parameters():
        assert param.grad is not None, f"Block 1 param {name} should have gradients."
        assert param.grad.abs().sum() > 0

    for name, param in model.blocks[0].named_parameters():
        assert param.grad is None, f"Block 0 param {name} should have NO gradients (isolated)."

    for name, param in model.blocks[2].named_parameters():
        assert param.grad is None, f"Block 2 param {name} should have NO gradients (isolated)."

def test_noprop_inference_forward():
    vocab_size = 64
    block_size = 10
    model = NoPropTinyGPT(vocab_size=vocab_size, block_size=block_size, n_layer=2, n_head=2, n_embd=16)
    model.eval()

    idx = torch.randint(1, vocab_size, (2, block_size))
    with torch.no_grad():
        logits, preds = model(idx)

    assert logits.shape == (2, block_size, vocab_size)
    assert len(preds) == 2
    assert preds[0].shape == (2, block_size, 16)
