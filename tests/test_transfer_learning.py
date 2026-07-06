from __future__ import annotations

import torch

from src.codonlm.model_tiny_gpt import TinyGPT
from src.codonlm.training.checkpoint import _load_transfer_state_dict


def test_transfer_state_dict_expands_vocab_by_token_name():
    source_itos = ["<PAD>", "<BOS_CDS>", "AAA", "TAA"]
    target_itos = ["<PAD>", "<BOS_CDS>", "<UTR_START>", "AAA", "TAA", "A"]

    source = TinyGPT(
        vocab_size=len(source_itos),
        block_size=8,
        n_layer=1,
        n_head=1,
        n_embd=8,
        dropout=0.0,
        tie_embeddings=False,
    )
    target = TinyGPT(
        vocab_size=len(target_itos),
        block_size=8,
        n_layer=1,
        n_head=1,
        n_embd=8,
        dropout=0.0,
        tie_embeddings=False,
    )

    with torch.no_grad():
        for idx in range(len(source_itos)):
            source.tok_emb.weight[idx].fill_(float(idx + 10))
            source.head.weight[idx].fill_(float(idx + 20))
        target.tok_emb.weight.fill_(-1.0)
        target.head.weight.fill_(-2.0)

    report = _load_transfer_state_dict(
        target,
        source.state_dict(),
        source_itos=source_itos,
        target_itos=target_itos,
    )

    assert "tok_emb.weight:4" in report["loaded_rows"]
    assert "head.weight:4" in report["loaded_rows"]
    assert torch.allclose(target.tok_emb.weight[3], torch.full((8,), 12.0))
    assert torch.allclose(target.tok_emb.weight[4], torch.full((8,), 13.0))
    assert torch.allclose(target.head.weight[3], torch.full((8,), 22.0))
    assert torch.allclose(target.head.weight[4], torch.full((8,), 23.0))
    assert torch.allclose(target.tok_emb.weight[2], torch.full((8,), -1.0))
    assert torch.allclose(target.tok_emb.weight[5], torch.full((8,), -1.0))
