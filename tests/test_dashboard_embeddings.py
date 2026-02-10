import pytest
import numpy as np
from unittest.mock import Mock
from src.eval.visualizer import Visualizer
from src.eval.aggregator import ResultsAggregator
import matplotlib.pyplot as plt

@pytest.fixture
def mock_aggregator():
    agg = Mock(spec=ResultsAggregator)
    emb_a = np.random.rand(100, 16)
    emb_b = np.random.rand(100, 16)
    agg.run_ids = ["run_a", "run_b"]
    
    def get_artifacts(run_id):
        if run_id == "run_a":
            return {"embeddings": emb_a}
        elif run_id == "run_b":
            return {"embeddings": emb_b}
        return {}
        
    agg.get_artifacts.side_effect = get_artifacts
    return agg

def test_visualizer_prepare_pca(mock_aggregator):
    viz = Visualizer(mock_aggregator)
    pca_data = viz.compute_pca(n_components=2)
    assert "run_a" in pca_data
    assert pca_data["run_a"].shape == (100, 2)

def test_visualizer_plot_pca(mock_aggregator):
    viz = Visualizer(mock_aggregator)
    fig = viz.plot_pca_comparison()
    assert isinstance(fig, plt.Figure)
    # Check if we have axes for each run
    assert len(fig.axes) >= 2
