import torch

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
