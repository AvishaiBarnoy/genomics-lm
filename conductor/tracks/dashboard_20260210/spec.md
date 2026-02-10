# Specification: Experiment Comparison Dashboard

## Overview
This track implements a high-level dashboard to aggregate and compare results from multiple model runs across the 6-step interpretability pipeline. The dashboard will provide a unified view of metrics, visualizations, and qualitative findings, facilitating rapid iteration and biological validation.

## Objectives
- Aggregate results from `outputs/scores/<RUN_ID>/` and `runs/<RUN_ID>/`.
- Provide a CLI-based or lightweight web-based interface for comparing multiple runs.
- Support all 6 steps of the interpretability pipeline:
    1. Frequencies
    2. Embeddings
    3. Attention
    4. Next-token Probes
    5. Saliency
    6. Linear Probes

## Functional Requirements
1. **Run Selection:** Allow users to specify multiple `RUN_IDs` for comparison.
2. **Metrics Aggregation:** Extract and tabulate primary metrics (PPL, AA probe accuracy, etc.) from `metrics.json`.
3. **Visual Comparison:** Display side-by-side or overlaid charts for embeddings (PCA/UMAP) and attention maps.
4. **Interpretability Summary:** Generate a summary report highlighting differences in motif detection and saliency spikes across runs.
5. **Export:** Support exporting the comparison summary as a Markdown or HTML report.

## Non-Functional Requirements
- **Performance:** Fast aggregation, even for dozens of runs.
- **Reproducibility:** Comparison results must be traceable back to the source `RUN_IDs`.
- **Maintainability:** Modular architecture to easily add new interpretability steps.

## User Interface (Draft)
- **Primary View:** A table showing core metrics for selected runs.
- **Detail View:** Drill-down into specific steps (e.g., Step 2: Embeddings) to see comparative plots.
- **CLI Command:** `python -m scripts.dashboard --runs run_a,run_b,run_c`
