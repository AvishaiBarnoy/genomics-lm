import streamlit as st
import os
import pandas as pd
from src.eval.aggregator import ResultsAggregator
from src.eval.visualizer import Visualizer

def get_available_runs(scores_dir="outputs/scores"):
    if not os.path.exists(scores_dir):
        return []
    return [d for d in os.listdir(scores_dir) if os.path.isdir(os.path.join(scores_dir, d)) and d != "compare"]

def main():
    st.set_page_config(page_title="Genomics-LM Dashboard", layout="wide")
    st.title("🧬 Genomics-LM Experiment Dashboard")

    st.sidebar.header("Run Selection")
    available_runs = get_available_runs()
    selected_runs = st.sidebar.multiselect("Select Runs to Compare", options=available_runs)

    if not selected_runs:
        st.warning("Please select at least one run from the sidebar.")
        return

    aggregator = ResultsAggregator(run_ids=selected_runs)
    
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
        
    except Exception as e:
        st.error(f"Error loading metrics: {e}")

if __name__ == "__main__":
    main()
