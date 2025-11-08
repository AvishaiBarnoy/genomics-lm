import torch
from src.codonlm.model_tiny_gpt import TinyGPT


def test_cross_entropy_float32_path_is_finite():
    torch.manual_seed(0)
    model = TinyGPT(vocab_size=68, block_size=16, n_layer=2, n_head=2, n_embd=16, dropout=0.0, use_checkpoint=False, label_smoothing=0.05)
    model.train(False)
    # Build a tiny batch with some pads (0) in targets
    B, T = 2, 12
    x = torch.randint(low=1, high=67, size=(B, T))
    y = x.clone()
    y[:, ::5] = 0  # inject pad tokens
    logits, loss = model(x, y)
    assert loss is not None
    assert torch.isfinite(loss), "Loss should be finite even with label_smoothing and pads"

