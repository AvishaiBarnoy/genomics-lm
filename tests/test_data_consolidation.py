import os
import json
import pytest
import numpy as np
import shutil
from pathlib import Path
from src.eval.aggregator import ResultsAggregator

def test_results_aggregator_path_resolution(tmp_path):
    # Setup directories
    runs_dir = tmp_path / "runs"
    legacy_scores_dir = tmp_path / "outputs" / "scores"
    
    runs_dir.mkdir(parents=True, exist_ok=True)
    legacy_scores_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Create a consolidated run (new layout)
    new_run_id = "test_run_new"
    new_run_path = runs_dir / new_run_id
    new_scores_path = new_run_path / "scores"
    new_scores_path.mkdir(parents=True, exist_ok=True)
    
    new_metrics = {"val_ppl": 12.3, "train_loss": 0.45}
    with open(new_scores_path / "metrics.json", "w") as f:
        json.dump(new_metrics, f)
        
    new_meta = {"status": "completed", "run_id": new_run_id}
    with open(new_run_path / "meta.json", "w") as f:
        json.dump(new_meta, f)
        
    with open(new_run_path / "log.txt", "w") as f:
        f.write("Line 1\nLine 2\n")
        
    # Create dummy artifacts.npz
    dummy_x = np.array([1, 2, 3])
    np.savez(new_run_path / "artifacts.npz", x=dummy_x)
    
    # 2. Create a legacy run (old layout)
    legacy_run_id = "test_run_legacy"
    legacy_run_path = runs_dir / legacy_run_id
    legacy_run_path.mkdir(parents=True, exist_ok=True)
    
    legacy_metrics = {"val_ppl": 15.6, "train_loss": 0.65}
    legacy_scores_run_path = legacy_scores_dir / legacy_run_id
    legacy_scores_run_path.mkdir(parents=True, exist_ok=True)
    with open(legacy_scores_run_path / "metrics.json", "w") as f:
        json.dump(legacy_metrics, f)
        
    # Instantiate aggregator
    aggregator = ResultsAggregator(
        run_ids=[new_run_id, legacy_run_id],
        scores_base_dir=str(legacy_scores_dir),
        runs_base_dir=str(runs_dir)
    )
    
    # Load metrics and assert correctness
    metrics = aggregator.load_metrics()
    assert new_run_id in metrics
    assert legacy_run_id in metrics
    
    assert metrics[new_run_id]["val_ppl"] == 12.3
    assert metrics[legacy_run_id]["val_ppl"] == 15.6
    
    # Check details load
    details = aggregator.get_run_details(new_run_id)
    assert details["meta"]["status"] == "completed"
    assert "Line 1" in details["log"]
    
    # Check artifacts load
    artifacts = aggregator.get_artifacts(new_run_id)
    assert np.array_equal(artifacts["x"], dummy_x)
