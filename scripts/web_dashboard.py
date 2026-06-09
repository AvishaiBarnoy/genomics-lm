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
    return [
        d
        for d in os.listdir(scores_dir)
        if os.path.isdir(os.path.join(scores_dir, d)) and d != "compare"
    ]


def prepare_pca_dataframe(pca_results):
    """Converts a dictionary of PCA results into a long-format DataFrame."""
    rows = []
    for run_id, data in pca_results.items():
        for i in range(data.shape[0]):
            rows.append(
                {
                    "Run ID": run_id,
                    "PC1": data[i, 0],
                    "PC2": data[i, 1],
                    "Point Index": i,
                }
            )
    return pd.DataFrame(rows)


def prepare_attention_dataframe(attention_data):
    """Converts a dictionary of attention entropy results into a long-format DataFrame."""
    rows = []
    for run_id, data in attention_data.items():
        for layer, entropy in enumerate(data):
            rows.append({"Run ID": run_id, "Layer": layer, "Entropy": entropy})
    return pd.DataFrame(rows)


def prepare_saliency_dataframe(saliency_data):
    """Converts a dictionary of saliency DataFrames into a long-format DataFrame."""
    all_rows = []
    for run_id, df in saliency_data.items():
        # df has columns [position, token, saliency]
        temp_df = df.copy()
        temp_df["Run ID"] = run_id
        temp_df = temp_df.rename(
            columns={"position": "Position", "saliency": "Saliency", "token": "Token"}
        )
        all_rows.append(temp_df)

    if not all_rows:
        return pd.DataFrame()
    return pd.concat(all_rows, ignore_index=True)


def main():
    st.set_page_config(page_title="Genomics-LM Dashboard", layout="wide")
    st.title("🧬 Genomics-LM Experiment Dashboard")

    st.sidebar.header("Run Selection")
    available_runs = get_available_runs()
    selected_runs = st.sidebar.multiselect(
        "Select Runs to Compare", options=available_runs
    )

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

    tab1, tab2, tab3 = st.tabs(
        ["🚀 Comparison", "🔍 Individual Run", "🧪 Model Playground"]
    )

    with tab1:
        try:
            metrics = aggregator.load_metrics()
            st.header("📊 Core Metrics")

            # Convert metrics to DataFrame for display
            rows = []
            for run_id, m in metrics.items():
                rows.append(
                    {
                        "Run ID": run_id,
                        "Val Loss": m.get("val_loss", m.get("best_val_loss", "N/A")),
                        "Perplexity": m.get("last_perplexity", m.get("val_ppl", "N/A")),
                        "Bio Recall": m.get("bio_recall", "N/A"),
                    }
                )
            df_metrics = pd.DataFrame(rows)
            st.table(df_metrics)

            st.divider()

            # New Stage 2 Section
            st.header("🧬 Stage 2: Biological Deep Dive")

            col_bio1, col_bio2 = st.columns(2)
            with col_bio1:
                st.subheader("📊 Biological Recall Score")
                st.info(
                    "Measures how many 'Known Motifs' (Start codons, Promoters) the AI discovered on its own."
                )
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
                    st.image(
                        dialect_img,
                        caption="Codon usage differences across Gram-positive, High-GC, and Entero groups.",
                    )
                else:
                    st.info("Run scripts/analyze_dialects.py to generate this view.")

            st.divider()
            col1, col2 = st.columns(2)

            with col1:
                st.header("📉 Step 2: Embeddings (PCA)")
                pca_results = visualizer.compute_pca()
                if pca_results:
                    df_pca = prepare_pca_dataframe(pca_results)
                    st.scatter_chart(df_pca, x="PC1", y="PC2", color="Run ID", size=10)
                else:
                    st.info("No embedding data available for PCA.")

            with col2:
                st.header("🧠 Step 3: Attention Entropy")
                attention_results = visualizer.compute_attention_entropy()
                if attention_results:
                    df_attn = prepare_attention_dataframe(attention_results)
                    st.line_chart(df_attn, x="Layer", y="Entropy", color="Run ID")
                else:
                    st.info("No attention data available.")

            st.divider()
            st.header("🔦 Step 5: Saliency Scores")
            st.info(
                "**What is this?** This is a 'Highlight Map'. It shows which specific positions in a gene the AI is most sensitive to. "
                "Peaks represent 'Hotspots' where even a small change in DNA would drastically change the model's prediction."
            )
            saliency_results = visualizer.load_saliency_data()
            if saliency_results:
                df_saliency = prepare_saliency_dataframe(saliency_results)
                st.line_chart(df_saliency, x="Position", y="Saliency", color="Run ID")
            else:
                st.info("No saliency data available.")

        except Exception as e:
            st.error(f"Error loading metrics: {e}")

    with tab2:
        st.header("🔎 Individual Run Details")
        selected_detail_run = st.selectbox(
            "Select Run to Inspect", options=selected_runs
        )

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
                        with st.spinner(
                            "Executing analysis pipeline... this may take a few minutes."
                        ):
                            import subprocess

                            # Use analysis.sh
                            cmd = f"PYTHONPATH=. ./analysis.sh {selected_detail_run} configs/tiny_mps.yaml"
                            res = subprocess.run(
                                cmd, shell=True, capture_output=True, text=True
                            )
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
                    summary_path = os.path.join(
                        "runs", selected_detail_run, "PLAIN_ENGLISH_SUMMARY.md"
                    )
                    if os.path.exists(summary_path):
                        with open(summary_path, "r") as f:
                            st.markdown(f.read())
                    else:
                        st.info("No Plain English summary generated yet.")

                    st.subheader("🧬 Structural Motif Audit")
                    audit_path = os.path.join(
                        "runs",
                        selected_detail_run,
                        "motif_mining",
                        "structural_motif_audit.json",
                    )
                    if os.path.exists(audit_path):
                        import json

                        with open(audit_path, "r") as f:
                            audit_data = json.load(f)
                            for m in audit_data:
                                if "structural_stats" in m:
                                    st.markdown(
                                        f"**{m['name']}** ({m.get('sequence', '')})"
                                    )
                                    st.write(
                                        f"Interpretations: {', '.join(m['interpretations'])}"
                                    )
                                    st.json(m["structural_stats"])
                    else:
                        st.info("No structural audit found.")

            except Exception as e:
                st.error(f"Error loading run details: {e}")

    with tab3:
        st.header("🧪 Model Playground")
        st.info(
            "Directly interact with trained language models and critics. Predict codon probabilities, generate sequences, or classify functional attributes."
        )

        # 1. Checkpoint discovery
        all_dirs = []
        if os.path.exists("runs"):
            all_dirs.extend(
                [
                    d
                    for d in os.listdir("runs")
                    if os.path.isdir(os.path.join("runs", d)) and d != "_summary"
                ]
            )
        if os.path.exists("outputs/checkpoints"):
            all_dirs.extend(
                [
                    d
                    for d in os.listdir("outputs/checkpoints")
                    if os.path.isdir(os.path.join("outputs/checkpoints", d))
                ]
            )
        all_dirs = sorted(list(set(all_dirs)))

        selected_model_run = st.selectbox(
            "Select Model Checkpoint", options=all_dirs, key="play_model"
        )
        task_mode = st.selectbox(
            "Task Mode",
            [
                "Next-Codon Prediction",
                "Sequence Generation",
                "Protein Attribute Critic",
            ],
            key="play_task",
        )

        if task_mode == "Protein Attribute Critic":
            st.subheader("Protein Attribute Critic (Multi-Task Classifier)")
            config_path = st.text_input("Config Path", "configs/protein_critic.yaml")
            default_ckpt = "runs/protein_critic/checkpoints/best_critic.pt"
            if not os.path.exists(default_ckpt):
                default_ckpt = "outputs/checkpoints/protein_critic/best_critic.pt"
            ckpt_path = st.text_input("Checkpoint Path", default_ckpt)
            vocabs_path = st.text_input(
                "Vocabs Path", "data/processed/protein_lm/multitask/task_vocabs.json"
            )

            aa_input = st.text_area("Amino Acid Sequence", "MQAPILVADLGSL")

            if st.button("Query Critic"):
                from src.eval.inference_playground import (
                    load_protein_classifier,
                    classify_protein,
                    PROTEIN_AVAILABLE,
                )

                if not PROTEIN_AVAILABLE:
                    st.error("ProteinLM dependencies or modules could not be imported.")
                else:
                    try:
                        with st.spinner("Loading Protein Critic..."):
                            model, tokenizer, itos_dict, device = (
                                load_protein_classifier(
                                    config_path, ckpt_path, vocabs_path
                                )
                            )
                        with st.spinner("Analyzing sequence..."):
                            results = classify_protein(
                                model, tokenizer, itos_dict, device, aa_input
                            )

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(
                                "Pfam Family",
                                results["family"]["prediction"],
                                f"{results['family']['probability'] * 100:.1f}% confidence",
                            )
                            st.write("**Top Choices:**")
                            for c in results["family"]["choices"]:
                                st.write(f"- {c['label']}: {c['prob'] * 100:.1f}%")
                        with col2:
                            st.metric(
                                "EC Function Class",
                                results["function"]["prediction"],
                                f"{results['function']['probability'] * 100:.1f}% confidence",
                            )
                            st.write("**Top Choices:**")
                            for c in results["function"]["choices"]:
                                st.write(f"- {c['label']}: {c['prob'] * 100:.1f}%")
                        with col3:
                            st.metric(
                                "Stability Category",
                                results["stability"]["prediction"],
                                f"{results['stability']['probability'] * 100:.1f}% confidence",
                            )
                            st.write("**Top Choices:**")
                            for c in results["stability"]["choices"]:
                                st.write(f"- {c['label']}: {c['prob'] * 100:.1f}%")
                    except Exception as e:
                        st.error(f"Error querying critic: {e}")

        elif selected_model_run:
            from src.eval.inference_playground import (
                load_codon_model,
                query_next_codon,
                generate_cds,
            )

            try:

                @st.cache_resource
                def get_cached_codon_model(run):
                    return load_codon_model(run)

                model, itos, stoi, device = get_cached_codon_model(selected_model_run)

                if task_mode == "Next-Codon Prediction":
                    st.subheader("Next-Codon Prediction")
                    dna_input = st.text_input(
                        "DNA Sequence Prefix (Space-separated codons or raw DNA string)",
                        "ATG GCT AAC",
                    )
                    top_k_next = st.slider(
                        "Top K Predictions", min_value=1, max_value=20, value=5
                    )

                    if st.button("Predict"):
                        with st.spinner("Predicting..."):
                            preds = query_next_codon(
                                model, stoi, itos, device, dna_input, top_k=top_k_next
                            )

                        if preds:
                            df_preds = pd.DataFrame(preds)
                            st.write("**Top Predictions:**")
                            st.bar_chart(df_preds, x="token", y="prob")
                            st.table(df_preds)
                        else:
                            st.warning("Invalid or empty input sequence.")

                elif task_mode == "Sequence Generation":
                    st.subheader("Sequence Generation")
                    dna_input = st.text_input("DNA Sequence Seed Prefix", "ATG")

                    st.sidebar.divider()
                    st.sidebar.subheader("🌡️ Generation Hyperparameters")
                    temp = st.sidebar.slider(
                        "Temperature", min_value=0.1, max_value=2.0, value=1.0, step=0.1
                    )
                    top_k_gen = st.sidebar.slider(
                        "Top-K Sampling", min_value=0, max_value=50, value=0
                    )
                    max_tokens = st.sidebar.slider(
                        "Max New Codons",
                        min_value=10,
                        max_value=1000,
                        value=150,
                        step=10,
                    )

                    col_opts1, col_opts2 = st.columns(2)
                    with col_opts1:
                        stop_eos = st.checkbox("Stop on <EOS_CDS>", value=True)
                    with col_opts2:
                        stop_bio = st.checkbox("Stop on Stop Codon", value=True)

                    if st.button("Generate"):
                        with st.spinner("Generating..."):
                            tokens, info = generate_cds(
                                model,
                                stoi,
                                itos,
                                device,
                                dna_prefix=dna_input,
                                max_new_tokens=max_tokens,
                                temperature=temp,
                                top_k=top_k_gen,
                                stop_on_eos=stop_eos,
                                stop_on_bio_stop=stop_bio,
                            )

                        st.subheader("Generated Sequence")
                        highlighted_html = []
                        for tok in tokens:
                            if tok == "<BOS_CDS>" or tok == "ATG":
                                highlighted_html.append(
                                    f"<span style='color: #4CAF50; font-weight: bold; border-bottom: 2px solid #4CAF50; padding: 2px;'>{tok}</span>"
                                )
                            elif tok in ["TAA", "TAG", "TGA", "<EOS_CDS>"]:
                                highlighted_html.append(
                                    f"<span style='color: #F44336; font-weight: bold; border-bottom: 2px solid #F44336; padding: 2px;'>{tok}</span>"
                                )
                            elif tok.startswith("<") and tok.endswith(">"):
                                highlighted_html.append(
                                    f"<span style='color: #2196F3; font-style: italic;'>{tok}</span>"
                                )
                            else:
                                highlighted_html.append(tok)

                        html_str = f"<div style='font-family: monospace; font-size: 1.1rem; line-height: 1.8; word-wrap: break-word; background-color: #1e1e1e; padding: 15px; border-radius: 8px; color: #d4d4d4;'>{' '.join(highlighted_html)}</div>"
                        st.markdown(html_str, unsafe_allow_html=True)

                        st.subheader("📊 Biological KPIs")
                        codon_list = [
                            t
                            for t in tokens
                            if len(t) == 3
                            and not (t.startswith("<") or t.endswith(">"))
                        ]
                        raw_seq = "".join(codon_list)

                        col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
                        with col_kpi1:
                            st.metric("Total Codons", len(codon_list))
                        with col_kpi2:
                            gc = (
                                (raw_seq.count("G") + raw_seq.count("C")) / len(raw_seq)
                                if raw_seq
                                else 0.0
                            )
                            st.metric("GC Content", f"{gc * 100:.1f}%")
                        with col_kpi3:
                            has_stop_only_at_end = False
                            stops = [
                                i
                                for i, t in enumerate(codon_list)
                                if t in ["TAA", "TAG", "TGA"]
                            ]
                            if stops:
                                has_stop_only_at_end = (
                                    stops[-1] == len(codon_list) - 1
                                ) and (len(stops) == 1)
                            st.metric(
                                "Valid ORF Structure",
                                "Yes" if has_stop_only_at_end else "No",
                                "No internal stops"
                                if has_stop_only_at_end
                                else f"{len(stops)} stops found",
                            )

                        # Automated Protein Critic Translation & Supervision
                        from src.eval.inference_playground import (
                            translate_codons_to_aa,
                            load_protein_classifier,
                            classify_protein,
                            PROTEIN_AVAILABLE,
                        )

                        aa_seq = translate_codons_to_aa(codon_list)

                        st.subheader("🧬 Translated Protein Sequence")
                        st.code(aa_seq, language="text")

                        if PROTEIN_AVAILABLE and aa_seq:
                            st.subheader("🛡️ Protein Critic (Supervisor Layer)")
                            critic_config = "configs/protein_critic.yaml"
                            critic_ckpt = (
                                "runs/protein_critic/checkpoints/best_critic.pt"
                            )
                            if not os.path.exists(critic_ckpt):
                                critic_ckpt = (
                                    "outputs/checkpoints/protein_critic/best_critic.pt"
                                )
                            critic_vocab = (
                                "data/processed/protein_lm/multitask/task_vocabs.json"
                            )

                            if (
                                os.path.exists(critic_config)
                                and os.path.exists(critic_ckpt)
                                and os.path.exists(critic_vocab)
                            ):
                                with st.spinner(
                                    "Running Protein Critic supervisor on generated sequence..."
                                ):
                                    try:
                                        c_model, c_tokenizer, c_itos, c_dev = (
                                            load_protein_classifier(
                                                critic_config, critic_ckpt, critic_vocab
                                            )
                                        )
                                        critic_results = classify_protein(
                                            c_model, c_tokenizer, c_itos, c_dev, aa_seq
                                        )

                                        col_c1, col_c2, col_c3 = st.columns(3)
                                        with col_c1:
                                            st.metric(
                                                "Pfam Family Target",
                                                critic_results["family"]["prediction"],
                                                f"{critic_results['family']['probability'] * 100:.1f}% prob",
                                            )
                                        with col_c2:
                                            st.metric(
                                                "EC Function Target",
                                                critic_results["function"][
                                                    "prediction"
                                                ],
                                                f"{critic_results['function']['probability'] * 100:.1f}% prob",
                                            )
                                        with col_c3:
                                            st.metric(
                                                "Thermodynamic Stability",
                                                critic_results["stability"][
                                                    "prediction"
                                                ],
                                                f"{critic_results['stability']['probability'] * 100:.1f}% prob",
                                            )
                                    except Exception as exc:
                                        st.warning(
                                            f"Could not run Protein Critic: {exc}"
                                        )
                            else:
                                st.info(
                                    "Protein Critic checkpoints or configs not found at default locations. Train a critic to enable online supervision."
                                )

            except Exception as e:
                st.error(f"Error loading model or running inference: {e}")


if __name__ == "__main__":
    main()
