# Implementation Plan: Motif Mining & Cluster Analysis

This plan outlines the steps to build the motif mining toolkit, following a test-driven approach.

## Phase 1: Embedding Extraction & Preprocessing [checkpoint: ]
Implement the logic to extract localized embeddings from trained models.

- [x] Task: Create `src/eval/motif_extractor.py` to handle subsequence extraction.
    - [x] Write Tests for sliding window extraction (overlapping vs non-overlapping).
    - [x] Implement sliding window extraction logic.
- [x] Task: Integrate model loading and hidden-state extraction.
    - [x] Write Tests for hidden-state capture from TinyGPT.
    - [x] Implement `extract_embeddings` function in `motif_extractor.py`.
- [x] Task: Conductor - User Manual Verification 'Phase 1: Embedding Extraction & Preprocessing' (Protocol in workflow.md)

## Phase 2: Modular Clustering Engine [checkpoint: ]
Implement the clustering logic using scikit-learn and hdbscan.

- [x] Task: Implement the `MotifClusterer` class with support for KMeans and HDBSCAN.
    - [x] Write Tests for clustering output consistency (same data -> same labels).
    - [x] Implement dimensionality reduction (PCA/UMAP) integration.
    - [x] Implement `cluster_embeddings` method.
- [x] Task: Create a CLI entry point for motif mining.
    - [x] Write Tests for CLI argument parsing and output directory creation.
    - [x] Implement `scripts/mine_motifs.py`.
- [x] Task: Conductor - User Manual Verification 'Phase 2: Modular Clustering Engine' (Protocol in workflow.md)

## Phase 3: Motif Analysis & Visualization [checkpoint: ]
Generate consensus representations and visualizations for discovered clusters.

- [x] Task: Implement consensus sequence and PWM generation.
    - [x] Write Tests for PWM calculation from a list of sequences.
    - [x] Implement `calculate_consensus` and `calculate_pwm` in `src/eval/motif_analysis.py`.
- [ ] Task: Integrate Sequence Logo generation using `logomaker`.
    - [ ] Write Tests for logo image generation.
    - [ ] Implement `save_sequence_logo` method.
- [x] Task: Implement summary report generation.
    - [x] Write Tests for Markdown report formatting.
    - [x] Implement `generate_motif_report` in `scripts/mine_motifs.py`.
- [x] Task: Conductor - User Manual Verification 'Phase 3: Motif Analysis & Visualization' (Protocol in workflow.md)
