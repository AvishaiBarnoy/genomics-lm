import pytest
import os
import json
from src.eval.aggregator import ResultsAggregator

@pytest.fixture
def mock_run_dir(tmp_path):
    run_dir = tmp_path / "runs" / "run_a"
    run_dir.mkdir(parents=True)
    
    meta = {"n_layer": 2, "n_head": 4}
    with open(run_dir / "meta.json", "w") as f:
        json.dump(meta, f)
        
    with open(run_dir / "log.txt", "w") as f:
        f.write("Line 1\nLine 2")
        
    return tmp_path

def test_aggregator_get_run_details(mock_run_dir):
    agg = ResultsAggregator(
        run_ids=["run_a"],
        runs_base_dir=str(mock_run_dir / "runs")
    )
    # This will fail because get_run_details doesn't exist
    try:
        details = agg.get_run_details("run_a")
        assert details["meta"]["n_layer"] == 2
        assert "Line 1" in details["log"]
    except AttributeError:
        pytest.fail("ResultsAggregator has no method get_run_details")