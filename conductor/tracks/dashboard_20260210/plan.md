# Implementation Plan: Experiment Comparison Dashboard

This plan outlines the steps to implement the experiment comparison dashboard, following a modular and test-driven approach.

## Phase 1: Data Aggregation & Core Metrics [checkpoint: 8b25ef1]
Focus on extracting and centralizing metrics from multiple runs.

- [x] Task: Design and implement the `ResultsAggregator` class to load data from `metrics.json` and `artifacts.npz`. a95d86f
- [x] Task: Create a CLI entry point for run selection and basic metric tabulation. 4e51ba3
- [x] Task: Conductor - User Manual Verification 'Phase 1: Data Aggregation & Core Metrics' (Protocol in workflow.md)

## Phase 2: Interpretability Step Integration (Steps 1-3) [checkpoint: 7084f22]
Integrate Frequencies, Embeddings, and Attention comparison logic.

- [x] Task: Implement comparative visualization for Step 2 (Embeddings). fb8bbdc
    - [x] Write Tests for embedding aggregation logic.
    - [x] Implement side-by-side PCA/UMAP plotting.
- [x] Task: Implement comparative visualization for Step 3 (Attention). 5583223
    - [x] Write Tests for attention head specialization comparison.
    - [x] Implement attention map comparison view.
- [x] Task: Conductor - User Manual Verification 'Phase 2: Interpretability Step Integration (Steps 1-3)' (Protocol in workflow.md)

## Phase 3: Interpretability Step Integration (Steps 4-6) & Export
Integrate Probes and Saliency, and implement report exporting.

- [x] Task: Implement comparative visualization for Step 5 (Saliency). ce95f55
    - [x] Write Tests for saliency spike alignment logic.
    - [x] Implement saliency comparison view.
- [ ] Task: Implement report export functionality (Markdown/HTML).
    - [ ] Write Tests for report generation and file writing.
    - [ ] Implement export to `outputs/reports/comparison_<TIMESTAMP>/`.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Interpretability Step Integration (Steps 4-6) & Export' (Protocol in workflow.md)
