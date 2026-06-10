# Interactive UI Playgrounds & Live Monitor Upgrades Specification

## Overview
Characterize the current Genomics-LM web dashboard and implement 5 core improvements to enhance its interactivity, real-time feedback, and interpretive visualization, transforming it into a high-impact demo for the HayaData 2026 conference.

## Current UI Characterization
The dashboard is a wide-layout Streamlit application ([web_dashboard.py](file:///Users/User/github/genomics-lm/scripts/web_dashboard.py)) with three core tabs:
- **Comparison:** Compares metrics, PCA embeddings, attention entropy, and saliency curves across runs.
- **Individual Run:** Lists hyperparameters, logs, plain summaries, and motif audits.
- **Model Playground:** Allows next-codon prediction, text-guided sequence generation (with start/stop highlights and online critic translation/scoring), and direct protein classification queries.

## Five Key Improvements
1. **Interactive Reset-and-Discard (ReD) Sampling Visualizer:** Add a toggle for ReD sampling under generation mode. Display a live log of stochastically discarded attempts, showing where they hit premature stop codons.
2. **Live Training Progress Monitor:** Create a status panel that auto-polls running jobs (reads active `runs/*/scores/curves.csv`), plotting real-time loss curves and training speeds (seq/sec) during active training.
3. **Interactive 3D DNA Shape Alignment Plot:** Render an interactive aligned chart (using Plotly/Altair) plotting physical properties (MGW, Roll, EP) directly aligned under each codon of the generated sequence.
4. **Attention Weight Heatmap Visualizer (Step 3):** Add a live attention heatmap component allowing users to inspect attention weights between layers and heads on customized prompt sequences.
5. **Synonymous Mutation Comparison View:** Allow users to align a generated sequence against a wild-type seed, visually highlighting synonymous substitutions and displaying their respective change in stability critic scores.
