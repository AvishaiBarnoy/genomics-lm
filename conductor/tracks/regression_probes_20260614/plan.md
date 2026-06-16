# Implementation Plan: Regression Probes for DNA Shape Decoding

## Phase 1: Implement the Regression Probing Script
Create the core regression script to decode 3D shapes from hidden states.

- [x] Task: Create `scripts/probe_structural_regression.py`.
    - Implement target generation for Minor Groove Width (MGW), Electrostatic Potential (EP), Roll, Propeller Twist (ProT), and Helix Twist (HelT).
    - Extract hidden representations from the model block outputs.
    - Implement a 5-fold cross-validated Ridge/Linear regression model.
    - Compute and print $R^2$ scores and Pearson correlation for all properties.
    - Merge results into the run's `metrics.json`.

## Phase 2: Evaluation Suite Integration
Integrate the new regression probe into the unified benchmarking suite.

- [x] Task: Update `scripts/evaluate_run.py`.
    - Modify the `--mode full` section to invoke `python -m scripts.probe_structural_regression --run_id <run_id>`.
    - Ensure it runs on CPU by default to prevent Apple Silicon MPS lockups.

## Phase 3: Validation, Testing, and Comparison
Write unit tests and compare baseline vs. transfer learning runs.

- [x] Task: Implement unit tests in `tests/test_structural_probe.py`.
    - Test the regression script end-to-end using a mock model and mock sequences.
    - Validate that shape targets and hidden dimensions align correctly.
- [x] Task: Run comparison.
    - Evaluate `2026-06-12_stage2.5_6L4H_d384_e5` and `2026-06-12_stage2.5_10L8H_d384_e5` models.
    - Compare unsupervised PCA(1) correlation vs. supervised regression $R^2$/correlation.
