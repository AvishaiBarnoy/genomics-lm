# Implementation Plan: Model Querying Streamlit UI

## Phase 1: Inference API Backend
Abstract the command-line query scripts into reusable helper functions.

- [ ] Task: Create `src/eval/inference_playground.py`.
    - Abstract next-codon probability calculation from `scripts/query_model.py`.
    - Abstract coding sequence generation (CDS generation with top_k and temperature sampling) from `scripts/infer_generate_cds.py`.
    - Abstract multi-task protein classifier querying from `scripts/eval_multi_task_critic.py`.

## Phase 2: Streamlit Dashboard Frontend
Integrate the frontend widgets and tabs.

- [ ] Task: Implement the "🔍 Model Playground" tab in `scripts/web_dashboard.py`.
    - Run ID selector: select from active models in `outputs/checkpoints/` (or the consolidated runs folder).
    - Task type selector: "Next-Codon", "Generate Sequence", "Classify Protein".
    - Input text area: DNA prefix sequence (or protein sequence for classification).
    - Hyperparameters sidebar: Temperature (0.1 - 2.0), Top-K (1 - 20), Max Tokens (32 - 1024).
- [ ] Task: Integrate prediction outputs & KPI metrics.
    - Render probabilities as a bar chart (Streamlit `st.bar_chart`).
    - Render generated sequences with high-contrast biological punctuation highlighting (highlighting Start/Stop codons in green/red).
    - Render biological evaluation KPIs (synonymous gap, stop-codon recall) dynamically when sequence generation is executed.

## Phase 3: Validation & Tests
Write unit tests and perform a visual check.

- [ ] Task: Write `tests/test_inference_playground.py`.
    - Test next-codon, sequence generation, and classification helpers with mock models to verify that the helper methods execute without error.
- [ ] Task: Pre-flight UI review.
    - Run `streamlit run scripts/web_dashboard.py` locally and verify the layout works correctly.
