# Specification: Regression Probes for DNA Shape Decoding

## Overview
Currently, our biophysical structural awareness check (`probe_structural_awareness.py`) maps the model's hidden states using an unsupervised 1D projection (PCA-1) and calculates Pearson correlation against Pentamer-based DNAshape parameters. While simple, this does not capture structural information encoded in secondary or tertiary dimensions of the hidden states, which are now heavily influenced by grammatical gene boundary and reading frame signals.

This track replaces/complements the PCA-1 projection with a supervised **Regression Probe** (Ridge Regression with 5-fold cross-validation) to decode continuous physical features (MGW, EP, Roll, ProT, HelT) directly from the model's high-dimensional representation space. This provides a more rigorous test of structural representation learning and allows a direct comparison between PCA-1 and full-space regression decoding.

## Objectives
1. **Supervised Structural Decoding**: Train Ridge/Linear regression models to map the model's $D$-dimensional hidden states to 3D DNAshape targets.
2. **Generalization Metrics**: Implement 5-fold cross-validation to report generalization $R^2$ scores and Pearson correlation on held-out test sequences, preventing leakage.
3. **Integration**: Add the regression probe into the unified evaluation suite (`scripts/evaluate_run.py`) under the `--mode full` configuration.
4. **Analysis & Comparison**: Compare the decoding capacity of the new model (trained on dynamic data) vs. older baselines to see if scaling dimensions and layer count improves structural awareness.

## Technical Details
* **Target Script**: `scripts/probe_structural_regression.py`
  - Loads model checkpoint and vocabulary.
  - Generates structural targets (MGW, EP, Roll, ProT, HelT) on test sequences.
  - Extracts hidden states from the model.
  - Fits a cross-validated Ridge regression model using `scikit-learn`.
  - Writes metrics ($R^2$ and correlation) to `metrics.json`.
* **Evaluation Integration**: Update `scripts/evaluate_run.py` to call `probe_structural_regression.py` as part of the full evaluation suite.
