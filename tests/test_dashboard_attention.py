import pytest
import numpy as np
from unittest.mock import Mock
from src.eval.visualizer import Visualizer
from src.eval.aggregator import ResultsAggregator
import matplotlib.pyplot as plt

@pytest.fixture
def mock_aggregator():
    agg = Mock(spec=ResultsAggregator)
    # attn shape: (layers, batch, heads, T, T)
    # 2 layers, 1 sample, 2 heads, 10 tokens
    attn_a = np.random.rand(2, 1, 2, 10, 10)
    # Normalize last dim to be probability-like
    attn_a = attn_a / attn_a.sum(axis=-1, keepdims=True)
    
    attn_b = np.random.rand(2, 1, 2, 10, 10)
    attn_b = attn_b / attn_b.sum(axis=-1, keepdims=True)
    
    agg.run_ids = ["run_a", "run_b"]
    
    def get_artifacts(run_id):
        if run_id == "run_a":
            return {"attn": attn_a}
        elif run_id == "run_b":
            return {"attn": attn_b}
        return {}
        
    agg.get_artifacts.side_effect = get_artifacts
    return agg

def test_visualizer_plot_attention_entropy(mock_aggregator):
    viz = Visualizer(mock_aggregator)
    fig = viz.plot_attention_entropy()
    assert isinstance(fig, plt.Figure)
    # Should have one ax
    assert len(fig.axes) == 1
