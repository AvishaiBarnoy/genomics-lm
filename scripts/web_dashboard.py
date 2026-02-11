import streamlit as st
import os
import sys
import pandas as pd

# Add project root to sys.path to resolve 'src' when running directly with streamlit
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.eval.aggregator import ResultsAggregator
from src.eval.visualizer import Visualizer

def get_available_runs(scores_dir="outputs/scores"):
    if not os.path.exists(scores_dir):
        return []
    return [d for d in os.listdir(scores_dir) if os.path.isdir(os.path.join(scores_dir, d)) and d != "compare"]

def prepare_pca_dataframe(pca_results):
    """Converts a dictionary of PCA results into a long-format DataFrame."""
    rows = []
    for run_id, data in pca_results.items():
        for i in range(data.shape[0]):
            rows.append({
                "Run ID": run_id,
                "PC1": data[i, 0],
                "PC2": data[i, 1],
                "Point Index": i
            })
    return pd.DataFrame(rows)

def prepare_attention_dataframe(attention_data):
    """Converts a dictionary of attention entropy results into a long-format DataFrame."""
    rows = []
    for run_id, data in attention_data.items():
        for layer, entropy in enumerate(data):
            rows.append({
                "Run ID": run_id,
                "Layer": layer,
                "Entropy": entropy
            })
    return pd.DataFrame(rows)

def prepare_saliency_dataframe(saliency_data):
    """Converts a dictionary of saliency DataFrames into a long-format DataFrame."""
    all_rows = []
    for run_id, df in saliency_data.items():
        # df has columns [position, token, saliency]
        temp_df = df.copy()
        temp_df["Run ID"] = run_id
        temp_df = temp_df.rename(columns={
            "position": "Position",
            "saliency": "Saliency",
            "token": "Token"
        })
        all_rows.append(temp_df)
    
    if not all_rows:
        return pd.DataFrame()
    return pd.concat(all_rows, ignore_index=True)

def main():
    st.set_page_config(page_title="Genomics-LM Dashboard", layout="wide")
    st.title("🧬 Genomics-LM Experiment Dashboard")

    st.sidebar.header("Run Selection")
    available_runs = get_available_runs()
    selected_runs = st.sidebar.multiselect("Select Runs to Compare", options=available_runs)

    if selected_runs:
        st.sidebar.divider()
        if st.sidebar.button("📄 Export Comparison Report"):
            aggregator = ResultsAggregator(run_ids=selected_runs)
            visualizer = Visualizer(aggregator)
            with st.spinner("Generating report..."):
                report_path = visualizer.export_report()
                st.sidebar.success(f"Report exported to: {report_path}")

    if not selected_runs:
        st.warning("Please select at least one run from the sidebar.")
        return

    aggregator = ResultsAggregator(run_ids=selected_runs)
    visualizer = Visualizer(aggregator)
    
    try:
        metrics = aggregator.load_metrics()
        st.header("📊 Core Metrics")
        
        # Convert metrics to DataFrame for display
        rows = []
        for run_id, m in metrics.items():
            rows.append({
                "Run ID": run_id,
                "Val Loss": m.get("val_loss", m.get("best_val_loss", "N/A")),
                "Perplexity": m.get("last_perplexity", "N/A")
            })
        df_metrics = pd.DataFrame(rows)
        st.table(df_metrics)

        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            st.header("📉 Step 2: Embeddings (PCA)")
            pca_results = visualizer.compute_pca()
            if pca_results:
                df_pca = prepare_pca_dataframe(pca_results)
                st.scatter_chart(
                    df_pca,
                    x="PC1",
                    y="PC2",
                    color="Run ID",
                    size=10
                )
            else:
                st.info("No embedding data available for PCA.")

        with col2:
            st.header("🧠 Step 3: Attention Entropy")
            attention_results = visualizer.compute_attention_entropy()
            if attention_results:
                df_attn = prepare_attention_dataframe(attention_results)
                st.line_chart(
                    df_attn,
                    x="Layer",
                    y="Entropy",
                    color="Run ID"
                )
            else:
                st.info("No attention data available.")

        st.divider()
        st.header("🔦 Step 5: Saliency Scores")
        saliency_results = visualizer.load_saliency_data()
        if saliency_results:
            df_saliency = prepare_saliency_dataframe(saliency_results)
            st.line_chart(
                df_saliency,
                x="Position",
                y="Saliency",
                color="Run ID"
            )
        else:
            st.info("No saliency data available.")
        
    except Exception as e:
        st.error(f"Error loading metrics: {e}")

if __name__ == "__main__":
    main()
