import pytest
import os
import json
import numpy as np
from src.eval.aggregator import ResultsAggregator

@pytest.fixture
def mock_run_dirs(tmp_path):
    # Create mock run directories
    run_a = tmp_path / "outputs" / "scores" / "run_a"
    run_a.mkdir(parents=True)
    metrics_a = {"run_id": "run_a", "val_loss": 1.0}
    with open(run_a / "metrics.json", "w") as f:
        json.dump(metrics_a, f)
    
    run_a_artifacts = tmp_path / "runs" / "run_a"
    run_a_artifacts.mkdir(parents=True)
    np.savez(run_a_artifacts / "artifacts.npz", data=np.array([1, 2, 3]))

    run_b = tmp_path / "outputs" / "scores" / "run_b"
    run_b.mkdir(parents=True)
    metrics_b = {"run_id": "run_b", "val_loss": 0.5}
    with open(run_b / "metrics.json", "w") as f:
        json.dump(metrics_b, f)

    run_b_artifacts = tmp_path / "runs" / "run_b"
    run_b_artifacts.mkdir(parents=True)
    np.savez(run_b_artifacts / "artifacts.npz", data=np.array([4, 5, 6]))

    return tmp_path

def test_aggregator_load_metrics(mock_run_dirs):
    agg = ResultsAggregator(
        run_ids=["run_a", "run_b"],
        scores_base_dir=str(mock_run_dirs / "outputs" / "scores"),
        runs_base_dir=str(mock_run_dirs / "runs")
    )
    metrics = agg.load_metrics()
    assert len(metrics) == 2
    assert metrics["run_a"]["val_loss"] == 1.0
    assert metrics["run_b"]["val_loss"] == 0.5

def test_aggregator_load_artifacts(mock_run_dirs):
    agg = ResultsAggregator(
        run_ids=["run_a", "run_b"],
        scores_base_dir=str(mock_run_dirs / "outputs" / "scores"),
        runs_base_dir=str(mock_run_dirs / "runs")
    )
    artifacts = agg.get_artifacts("run_a")
    assert np.array_equal(artifacts["data"], np.array([1, 2, 3]))

def test_aggregator_invalid_run(mock_run_dirs):
    agg = ResultsAggregator(
        run_ids=["run_invalid"],
        scores_base_dir=str(mock_run_dirs / "outputs" / "scores"),
        runs_base_dir=str(mock_run_dirs / "runs")
    )
    # Should now return an empty dict instead of raising FileNotFoundError
    metrics = agg.load_metrics()
    assert metrics == {}

