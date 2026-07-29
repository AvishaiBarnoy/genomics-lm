import torch

from scripts.diagnose_termination_probabilities import (
    _summarize,
    _token_probabilities,
)


class FixedLogitModel(torch.nn.Module):
    block_size = 8

    def forward(self, token_ids):
        logits = torch.zeros(
            token_ids.shape[0], token_ids.shape[1], 6, device=token_ids.device
        )
        logits[..., 4] = 2.0
        logits[..., 5] = 1.0
        return logits, None


def test_token_probability_diagnostic_reports_ranks_and_summary():
    rows = _token_probabilities(
        FixedLogitModel(),
        [1, 2, 3],
        [("final", 2)],
        stop_ids=[4],
        eos_id=5,
        device=torch.device("cpu"),
    )

    assert len(rows) == 1
    assert rows[0]["best_termination_rank"] == 1
    assert rows[0]["termination_in_top5"] is True
    summary = _summarize(rows)
    assert summary["final"]["n"] == 1
    assert summary["final"]["mean_termination_probability"] > 0.5
