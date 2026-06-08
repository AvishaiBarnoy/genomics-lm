# Specification: Motif Mining & Cluster Analysis

## Overview
This track implements Step 3 of the Roadmap Project: "Motif Mining". It provides a toolkit to extract hidden-state embeddings, cluster subsequences using multiple algorithms, and visualize the resulting biological motifs. This tool bridges the gap between deep learning representations and biological interpretation.

## Objectives
- Extract localized hidden states from a trained model for any given DNA/mRNA sequence.
- Support multiple clustering algorithms (K-Means, HDBSCAN, DBSCAN) to find recurring patterns.
- Generate biological consensus representations (Sequence Logos, PWMs) for discovered motifs.

## Functional Requirements
1. **Embedding Extraction:**
    - Load a trained model checkpoint.
    - Support three extraction modes:
        - **Fixed-Length Sliding Window:** Configurable window size and stride.
        - **Variable-Length Segments:** Extraction based on saliency or attention spikes.
        - **Whole-Sequence:** Average pooling of hidden states across the sequence.
2. **Clustering Engine:**
    - Implement a modular interface for `KMeans`, `HDBSCAN`, and `DBSCAN` (using `scikit-learn` and `hdbscan` libraries).
    - Provide dimensionality reduction (PCA or UMAP) before clustering to improve performance.
3. **Motif Analysis:**
    - Calculate **Centroid Representatives** for each cluster.
    - Generate **Position Weight Matrices (PWMs)** and **Sequence Logos** for clusters.
    - (Optional) Compute enrichment stats vs. known annotations.
4. **Visualization:**
    - Plot clusters in a 2D/3D space using UMAP/PCA.
    - Export a summary report (`motif_report.md`) with cluster sizes and top motifs.

## Technical Details
- **Libraries:** `scikit-learn`, `hdbscan`, `logomaker` (for sequence logos).
- **Output:** `runs/<RUN_ID>/motif_mining/` containing `clusters.npz`, `report.md`, and visualizations.

## Acceptance Criteria
- Successfully extract and cluster 10,000 windows from a validation set.
- Generate at least one valid Sequence Logo image for a high-silhouette cluster.
- Identify at least one cluster that corresponds to a known start/stop or biological pattern.
