import pytest
import pandas as pd
import io
import os
from unittest.mock import Mock, patch
from src.eval.visualizer import Visualizer
from src.eval.aggregator import ResultsAggregator
import matplotlib.pyplot as plt

@pytest.fixture
def mock_aggregator():
    agg = Mock(spec=ResultsAggregator)
    agg.run_ids = ["run_a", "run_b"]
    agg.runs_base_dir = "runs"
    return agg

def test_visualizer_plot_saliency(mock_aggregator):
    viz = Visualizer(mock_aggregator)
    
    # Mock reading CSV files
    csv_content_a = "position,token,saliency\n0,ATG,0.1\n1,GCT,0.2"
    csv_content_b = "position,token,saliency\n0,ATG,0.05\n1,GCT,0.3"
    
    def mock_exists(path):
        return True
    
    def mock_open(path, mode='r', **kwargs):
        if "run_a" in str(path):
            return io.StringIO(csv_content_a)
        else:
            return io.StringIO(csv_content_b)

    with patch("os.path.exists", side_effect=mock_exists):
        with patch("builtins.open", side_effect=mock_open):
            fig = viz.plot_saliency_comparison()
            assert isinstance(fig, plt.Figure)
            assert len(fig.axes) == 1