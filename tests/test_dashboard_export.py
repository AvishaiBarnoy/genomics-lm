import pytest
import os
import shutil
from unittest.mock import Mock, patch
from src.eval.visualizer import Visualizer
from src.eval.aggregator import ResultsAggregator

@pytest.fixture
def mock_aggregator():
    agg = Mock(spec=ResultsAggregator)
    agg.run_ids = ["run_a", "run_b"]
    agg.metrics = {
        "run_a": {"val_loss": 1.0, "last_perplexity": 50.0},
        "run_b": {"val_loss": 0.5, "last_perplexity": 40.0}
    }
    agg.load_metrics.return_value = agg.metrics
    return agg

def test_visualizer_export_report(mock_aggregator, tmp_path):
    viz = Visualizer(mock_aggregator)
    
    # Mocking compute_pca and other plot methods to avoid heavy lifting
    with patch.object(Visualizer, 'compute_pca', return_value={}):
        with patch.object(Visualizer, 'plot_pca_comparison') as mock_plot_pca:
            with patch.object(Visualizer, 'plot_attention_entropy') as mock_plot_attn:
                with patch.object(Visualizer, 'plot_saliency_comparison') as mock_plot_saliency:
                    
                    report_dir = tmp_path / "reports"
                    viz.export_report(output_dir=str(report_dir))
                    
                    assert report_dir.exists()
                    assert (report_dir / "report.md").exists()
                    # Check if images are saved (if we implement that)
                    # assert (report_dir / "pca_comparison.png").exists()
