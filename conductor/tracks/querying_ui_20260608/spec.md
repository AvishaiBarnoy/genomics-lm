# Specification: Model Querying Streamlit UI

## Overview
Currently, running model queries, generating CDS sequences, or classifying sequences with the trained model requires executing separate command-line python scripts (e.g., `scripts/query_model.py`, `scripts/infer_generate_cds.py`, `src/protein_lm/evaluate_classifier.py`).

This track integrates an interactive **Model Playground** tab directly into the existing Streamlit dashboard (`scripts/web_dashboard.py`). It enables users to visually query models, customize generation parameters (temperature, top_k, length), and observe model predictions and biophysical KPI evaluations in real time.

## Objectives
1. **Interactive Query Interface**: Add a tab in the web dashboard for user queries.
2. **Support Multiple Modes**:
   - **Next-Codon Prediction**: Predict codon probabilities given a DNA prefix.
   - **Sequence Generation**: Generate complete coding sequences (CDS) given a starter prefix, including real-time biological KPI validation.
   - **Protein Classifier**: Predict Pfam family, EC function, and stability for a given protein sequence.
3. **Parameter Controls**: Provide simple UI widgets (sliders, text inputs) for `temperature`, `top_k`, and `max_tokens`.

## Technical Details
- **Dashboard File**: `scripts/web_dashboard.py` (add Streamlit components).
- **Backend Utilities**: Reuse the inference and tokenization logic in `scripts/query_model.py` and `scripts/infer_generate_cds.py` by abstracting them into a clean inference helper class.
