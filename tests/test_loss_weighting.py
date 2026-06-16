import torch
from src.codonlm.model_tiny_gpt import TinyGPT


def test_tinygpt_loss_weighting():
    vocab, T, B = 10, 5, 2

    # Standard model (uniform loss weights)
    m_std = TinyGPT(vocab_size=vocab, block_size=T, n_layer=1, n_head=1, n_embd=16, dropout=0.0, use_checkpoint=False)

    # Weighted model (upweight token 5 by 10x)
    loss_weights = [1.0] * vocab
    loss_weights[5] = 10.0
    m_weighted = TinyGPT(
        vocab_size=vocab, block_size=T, n_layer=1, n_head=1, n_embd=16, dropout=0.0, use_checkpoint=False,
        loss_weights=loss_weights
    )

    # Share weights/biases to make outputs identical before weighting
    sd = {k: v for k, v in m_std.state_dict().items() if k != "loss_weights"}
    m_weighted.load_state_dict(sd, strict=False)

    print("m_std weights:", m_std.loss_weights)
    print("m_weighted weights:", m_weighted.loss_weights)

    # Input tokens
    x = torch.randint(1, vocab, (B, T), dtype=torch.long)

    # Case A: Targets without token 5
    y_no_5 = torch.ones((B, T), dtype=torch.long)  # all token 1

    # Case B: Mixed targets containing both standard token 1 and upweighted token 5
    y_mix = torch.tensor([[1, 5, 1, 5, 1], [1, 5, 1, 5, 1]], dtype=torch.long)

    # Forward pass on standard model
    _, loss_std_no_5 = m_std(x, y_no_5)
    _, loss_std_mix = m_std(x, y_mix)

    # Forward pass on weighted model
    _, loss_weighted_no_5 = m_weighted(x, y_no_5)
    _, loss_weighted_mix = m_weighted(x, y_mix)

    # Since y_no_5 has only token 1 (weight 1.0), standard loss and weighted loss should be extremely close
    assert torch.allclose(loss_std_no_5, loss_weighted_no_5, atol=1e-4)

    # Since y_mix contains token 5 (weighted 10x), the weighted loss and standard loss must differ
    print(f"loss_std_mix: {loss_std_mix.item():.4f}, loss_weighted_mix: {loss_weighted_mix.item():.4f}")
    assert not torch.allclose(loss_std_mix, loss_weighted_mix, atol=1e-4)
