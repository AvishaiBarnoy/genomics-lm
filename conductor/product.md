# Initial Concept
A compact codon-level GPT-style LM with a reproducible training + analysis pipeline.

# Product Definition

## Target Audience
The project is specifically designed for **Machine Learning Researchers** who are focused on genomic data. It provides the tools and framework necessary to apply advanced language modeling techniques to biological sequences.

## Primary Goals
1.  **Biological Pattern Discovery:** Develop highly interpretable models that can reveal hidden biological motifs and structural patterns within genomic sequences.
2.  **Accessible Experimentation:** Create a fast, efficient, and reproducible training pipeline that allows for rapid experimentation and model development even on standard consumer-grade hardware.

## Key Features
-   **6-Step Interpretability Pipeline:** An automated suite of analysis tools (including embedding visualization, attention specializations, and saliency maps) specifically designed for biological validation of model outputs.
-   **Experiment Comparison Dashboard:** A high-level dashboard to aggregate and compare results from multiple model runs, featuring metric tabulation, side-by-side visualizations (PCA, Attention Entropy, Saliency), and report exporting.
-   **Modular Agentic Architecture:** A structured development approach using conceptual "agents" to handle discrete tasks like data extraction, tokenization, and multi-stage model training.
-   **Motif & Mutation Analysis:** Dedicated tools for generating mutation heatmaps and mining functional motifs directly from the model's hidden states to understand sequence-function relationships.
