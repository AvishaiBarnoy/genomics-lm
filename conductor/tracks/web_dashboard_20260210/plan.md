# Implementation Plan: Web-based Dashboard GUI

This plan outlines the steps to build the Streamlit-based dashboard.

## Phase 1: Streamlit Setup & Metrics [checkpoint: 79b2a64]
Establish the basic web application structure.

- [x] Task: Install Streamlit and create the basic app skeleton. f529f9b
- [x] Task: Conductor - User Manual Verification 'Phase 1: Streamlit Setup & Metrics' (Protocol in workflow.md)

## Phase 2: Interactive Visualizations [checkpoint: ]
Integrate PCA and Attention plots into the Streamlit UI.

- [ ] Task: Integrate PCA visualization with interactive controls.
    - [ ] Write Tests for embedding data formatting for Streamlit.
    - [ ] Implement PCA plot in `web_dashboard.py`.
- [ ] Task: Integrate Attention Entropy visualization.
    - [ ] Write Tests for attention data formatting for Streamlit.
    - [ ] Implement Attention plot in `web_dashboard.py`.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Interactive Visualizations' (Protocol in workflow.md)

## Phase 3: Saliency & Export Integration [checkpoint: ]
Add saliency charts and export triggers.

- [ ] Task: Integrate Saliency visualization.
    - [ ] Write Tests for saliency data formatting for Streamlit.
    - [ ] Implement Saliency plot in `web_dashboard.py`.
- [ ] Task: Implement Export Report button.
    - [ ] Write Tests for report trigger logic.
    - [ ] Implement the export button in the sidebar.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Saliency & Export Integration' (Protocol in workflow.md)

## Phase 4: Individual Run Investigation [checkpoint: ]
Provide a detailed view for a single selected run.

- [ ] Task: Implement a "Run Details" page/mode in Streamlit.
    - [ ] Write Tests for run-specific data loading.
    - [ ] Implement hyperparameter and log display for the selected run.
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Individual Run Investigation' (Protocol in workflow.md)
