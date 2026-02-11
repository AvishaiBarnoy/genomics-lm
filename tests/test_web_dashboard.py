import pytest
from unittest.mock import patch, MagicMock
import sys

@pytest.fixture
def mock_streamlit():
    with patch('streamlit.sidebar') as mock_sidebar, \
         patch('streamlit.title') as mock_title, \
         patch('streamlit.multiselect') as mock_multiselect, \
         patch('streamlit.write') as mock_write, \
         patch('streamlit.table') as mock_table, \
         patch('streamlit.set_page_config') as mock_config:
        yield {
            'sidebar': mock_sidebar,
            'title': mock_title,
            'multiselect': mock_multiselect,
            'write': mock_write,
            'table': mock_table,
            'set_page_config': mock_config
        }

def test_web_dashboard_layout(mock_streamlit):
    with patch('scripts.web_dashboard.ResultsAggregator') as mock_agg:
        from scripts.web_dashboard import main
        
        # Mock aggregator behavior
        mock_agg_instance = mock_agg.return_value
        mock_agg_instance.load_metrics.return_value = {
            "run_a": {"val_loss": 1.0, "last_perplexity": 50.0}
        }
        
        # Mock streamlit interaction
        mock_streamlit['sidebar'].multiselect.return_value = ["run_a"]
        
        main()
        
        mock_streamlit['title'].assert_called()
        mock_streamlit['sidebar'].multiselect.assert_called()
        mock_streamlit['table'].assert_called()