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
    
    tab1, tab2 = st.tabs(["🚀 Comparison", "🔍 Individual Run"])

    with tab1:
        try:
            metrics = aggregator.load_metrics()
            st.header("📊 Core Metrics")
            
            # Convert metrics to DataFrame for display
            rows = []
            for run_id, m in metrics.items():
                rows.append({
                    "Run ID": run_id,
                    "Val Loss": m.get("val_loss", m.get("best_val_loss", "N/A")),
                    "Perplexity": m.get("last_perplexity", m.get("val_ppl", "N/A")),
                    "Bio Recall": m.get("bio_recall", "N/A")
                })
            df_metrics = pd.DataFrame(rows)
            st.table(df_metrics)

            st.divider()
            
            # New Stage 2 Section
            st.header("🧬 Stage 2: Biological Deep Dive")
            
            col_bio1, col_bio2 = st.columns(2)
            with col_bio1:
                st.subheader("📊 Biological Recall Score")
                st.info("Measures how many 'Known Motifs' (Start codons, Promoters) the AI discovered on its own.")
                # Bar chart of Bio Recall
                if not df_metrics.empty and "Bio Recall" in df_metrics.columns:
                    # Clean N/A for plotting
                    plot_df = df_metrics[df_metrics["Bio Recall"] != "N/A"].copy()
                    if not plot_df.empty:
                        plot_df["Bio Recall"] = plot_df["Bio Recall"].astype(float)
                        st.bar_chart(plot_df, x="Run ID", y="Bio Recall")
            
            with col_bio2:
                st.subheader("🗣️ Genomic Dialects")
                dialect_img = "outputs/analysis/stage2_diversity/dialect_comparison.png"
                if os.path.exists(dialect_img):
                    st.image(dialect_img, caption="Codon usage differences across Gram-positive, High-GC, and Entero groups.")
                else:
                    st.info("Run scripts/analyze_dialects.py to generate this view.")

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
            st.info("**What is this?** This is a 'Highlight Map'. It shows which specific positions in a gene the AI is most sensitive to. "
                    "Peaks represent 'Hotspots' where even a small change in DNA would drastically change the model's prediction.")
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

    with tab2:
        st.header("🔎 Individual Run Details")
        selected_detail_run = st.selectbox("Select Run to Inspect", options=selected_runs)
        
        if selected_detail_run:
            try:
                details = aggregator.get_run_details(selected_detail_run)
                
                col_m, col_l = st.columns([1, 2])
                
                with col_m:
                    st.subheader("📋 Hyperparameters")
                    if details["meta"]:
                        st.json(details["meta"])
                    else:
                        st.info("No metadata found.")
                    
                    st.divider()
                    st.subheader("🛠️ Operations")
                    if st.button(f"🚀 Run Full Analysis for {selected_detail_run}"):
                        with st.spinner("Executing analysis pipeline... this may take a few minutes."):
                            import subprocess
                            # Use analysis.sh
                            cmd = f"PYTHONPATH=. ./analysis.sh {selected_detail_run} configs/tiny_mps.yaml"
                            res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                            if res.returncode == 0:
                                st.success("Analysis complete! Refresh to see results.")
                            else:
                                st.error(f"Analysis failed: {res.stderr}")

                with col_l:
                    st.subheader("📄 Training Log")
                    if details["log"]:
                        st.text_area("Full Log Output", details["log"], height=300)
                    else:
                        st.info("No log file found.")
                    
                    st.subheader("📝 Biological Summary")
                    summary_path = os.path.join("runs", selected_detail_run, "PLAIN_ENGLISH_SUMMARY.md")
                    if os.path.exists(summary_path):
                        with open(summary_path, "r") as f:
                            st.markdown(f.read())
                    else:
                        st.info("No Plain English summary generated yet.")
                    
                    st.subheader("🧬 Structural Motif Audit")
                    audit_path = os.path.join("runs", selected_detail_run, "motif_mining", "structural_motif_audit.json")
                    if os.path.exists(audit_path):
                        import json
                        with open(audit_path, "r") as f:
                            audit_data = json.load(f)
                            for m in audit_data:
                                if "structural_stats" in m:
                                    st.markdown(f"**{m['name']}** ({m.get('sequence', '')})")
                                    st.write(f"Interpretations: {', '.join(m['interpretations'])}")
                                    st.json(m["structural_stats"])
                    else:
                        st.info("No structural audit found.")
                        
            except Exception as e:
                st.error(f"Error loading run details: {e}")

if __name__ == "__main__":
    main()
