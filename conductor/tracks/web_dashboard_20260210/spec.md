# Specification: Web-based Dashboard GUI

## Overview
This track implements a graphical user interface (GUI) for the experiment comparison dashboard using Streamlit. It will provide an interactive way to select runs, view metrics, and inspect visualizations (PCA, Attention, Saliency) side-by-side.

## Objectives
- Replace/Extend the CLI dashboard with a user-friendly web interface.
- Leverage `src/eval/aggregator.py` and `src/eval/visualizer.py` for backend logic.
- Provide interactive controls for filtering and exploring model runs.

## Functional Requirements
1. **Interactive Run Selection:** A sidebar or multi-select dropdown to choose `RUN_IDs`.
2. **Metrics Table:** An interactive table showing core metrics with sorting/filtering.
3. **Dynamic Visualizations:**
    - PCA plot with adjustable components (if applicable).
    - Attention Entropy plots.
    - Saliency score line charts.
4. **Live Inspection:** Ability to hover over data points in plots to see specific details (run name, position, etc.).
5. **Report Trigger:** A button to trigger the `export_report` logic and provide a download link or path.

## Non-Functional Requirements
- **Responsiveness:** The dashboard should update quickly when new runs are selected.
- **Ease of Use:** Minimal setup (one command to launch).
- **Extensibility:** Easy to add new Streamlit widgets for new interpretability steps.

## Technical Choice
- **Framework:** Streamlit (Python-native, excellent for ML dashboards).
- **Entry Point:** `scripts/web_dashboard.py`
