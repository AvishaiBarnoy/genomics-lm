import torch

from src.codonlm.model_tiny_gpt import TinyGPT, Cfg as CfgV1
from src.codonlm.model_tiny_gpt_v2 import TinyGPTv2


def test_tinygpt_v1_forward_shapes():
    vocab, T, B = 69, 16, 4
    cfg = CfgV1(vocab, n_layer=1, n_head=1, n_embd=32, block_size=T, dropout=0.0)
    m = TinyGPT(cfg)
    x = torch.randint(0, vocab, (B, T), dtype=torch.long)
    y = x.clone()
    logits, loss = m(x, y)
    assert logits.shape == (B, T, vocab)
    assert loss is not None and torch.isfinite(loss)


def test_tinygpt_v2_forward_shapes():
    vocab, T, B = 69, 16, 4
    m = TinyGPTv2(vocab_size=vocab, block_size=T, n_layer=1, n_head=1, n_embd=32, dropout=0.0, use_checkpoint=False)
    x = torch.randint(0, vocab, (B, T), dtype=torch.long)
    y = x.clone()
    y[:, 0] = 0  # pad to exercise ignore_index=0 in loss
    logits, loss = m(x, y)
    assert logits.shape == (B, T, vocab)
    assert loss is not None and torch.isfinite(loss)

