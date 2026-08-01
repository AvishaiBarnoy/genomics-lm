import pytest
import torch

from scripts.evaluate_termination_head import summarize_confusion


def test_summarize_confusion_reports_class_imbalance():
    confusion = torch.tensor([[8, 2], [9, 81]], dtype=torch.long)
    probability_sums = torch.tensor([6.0, 63.0], dtype=torch.float64)

    result = summarize_confusion(confusion, probability_sums)

    assert result["evaluated_positions"] == 100
    assert result["accuracy"] == pytest.approx(0.89)
    assert result["balanced_accuracy"] == pytest.approx(0.85)
    assert result["classes"][0]["recall"] == pytest.approx(0.8)
    assert result["classes"][1]["recall"] == pytest.approx(0.9)
    assert result["classes"][0]["mean_true_probability"] == pytest.approx(0.6)
    assert result["classes"][1]["mean_true_probability"] == pytest.approx(0.7)
