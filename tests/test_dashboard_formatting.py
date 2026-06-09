import pandas as pd
import numpy as np
from scripts.web_dashboard import prepare_pca_dataframe

def test_prepare_pca_dataframe():
    pca_results = {
        "run_a": np.array([[1, 2], [3, 4]]),
        "run_b": np.array([[5, 6], [7, 8]])
    }
    df = prepare_pca_dataframe(pca_results)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4
    assert set(df["Run ID"].unique()) == {"run_a", "run_b"}
    assert "PC1" in df.columns
    assert "PC2" in df.columns

def test_prepare_attention_dataframe():
    # Visualizer.plot_attention_entropy logic calculates avg_entropy per layer
    # For streamlit, we might want a DataFrame with [Layer, Run ID, Entropy]
    attention_data = {
        "run_a": np.array([0.1, 0.2, 0.3]),
        "run_b": np.array([0.15, 0.25, 0.35])
    }
    from scripts.web_dashboard import prepare_attention_dataframe
    df = prepare_attention_dataframe(attention_data)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 6
    assert "Layer" in df.columns
    assert "Entropy" in df.columns
    assert "Run ID" in df.columns

def test_prepare_saliency_dataframe():
    # Saliency results from multiple runs
    saliency_data = {
        "run_a": pd.DataFrame({"position": [0, 1], "token": ["A", "G"], "saliency": [0.1, 0.2]}),
        "run_b": pd.DataFrame({"position": [0, 1], "token": ["A", "G"], "saliency": [0.05, 0.25]})
    }
    from scripts.web_dashboard import prepare_saliency_dataframe
    df = prepare_saliency_dataframe(saliency_data)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4
    assert "Position" in df.columns
    assert "Saliency" in df.columns
    assert "Run ID" in df.columns


