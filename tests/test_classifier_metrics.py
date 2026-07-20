import numpy as np
from src.classifiers.probes import _stratified_bootstrap_indices, compute_metrics

def test_compute_metrics_imbalanced_and_bootstrap():
    # Mock data: Binary classification with 10 samples
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0])  # 2 errors (1 FP, 2 FN) -> Accuracy = 70%
    y_proba = np.array([0.1, 0.2, 0.15, 0.3, 0.7, 0.9, 0.85, 0.8, 0.4, 0.35])  # probabilities for positive class

    # Calculate metrics with bootstrapping
    metrics = compute_metrics(y_true, y_pred, y_proba, bootstrap=True, n_resamples=100, seed=42)
    
    # 1. Point estimate assertions
    assert "accuracy" in metrics
    assert "balanced_accuracy" in metrics
    assert "macro_f1" in metrics
    assert "auroc" in metrics
    assert "macro_auprc" in metrics
    
    assert metrics["accuracy"] == 0.7
    assert metrics["balanced_accuracy"] == 0.7
    
    # 2. Bootstrap CI assertions
    assert "accuracy_ci_lower" in metrics
    assert "accuracy_ci_upper" in metrics
    assert "balanced_accuracy_ci_lower" in metrics
    assert "balanced_accuracy_ci_upper" in metrics
    assert "macro_auprc_ci_lower" in metrics
    
    assert 0.0 <= metrics["accuracy_ci_lower"] <= metrics["accuracy"]
    assert metrics["accuracy"] <= metrics["accuracy_ci_upper"] <= 1.0
    
    print("Metrics and bootstrap CIs validated successfully.")


def test_stratified_bootstrap_preserves_each_class_count():
    y_true = np.array([0, 0, 0, 1, 1, 2])
    indices = _stratified_bootstrap_indices(y_true, np.random.default_rng(7))
    assert np.bincount(y_true[indices]).tolist() == [3, 2, 1]


def test_stratified_bootstrap_intervals_are_deterministic():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    first = compute_metrics(y_true, y_pred, bootstrap=True, n_resamples=30, seed=9)
    second = compute_metrics(y_true, y_pred, bootstrap=True, n_resamples=30, seed=9)
    assert first == second
