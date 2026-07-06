# Biophysical Regression & MLP Probes Plan

This track captures the design and implementation of advanced probing techniques to evaluate CodonLM static token embeddings. It introduces continuous regression probes targeting biophysical properties (hydropathy, volume, charge) and non-linear MLP classifiers to handle codon degeneracy mapping.

---

## Status

- **State:** Completed
- **Completed:** 2026-07-06
- **Owner:** conductor
- **Primary files:**
  - [probe_linear.py](file:///Users/User/github/genomics-lm/scripts/probe_linear.py)
  - [generate_probe_labels.py](file:///Users/User/github/genomics-lm/scripts/generate_probe_labels.py)
- **Risk level:** Low. Only runs evaluation probes on existing checkpoints; does not modify model training loops.

---

## Design Principles

- **Continuous Targets:** High-cardinality classification (like 20-class amino acid identity) is sparse when trained on only 64 static codon embeddings. Mapping codons to continuous biophysical properties (such as Kyte-Doolittle hydropathy scales) reduces the target space to a robust 1D regression problem.
- **Non-Linear Probing (MLPs):** Codon-to-amino-acid mappings have non-linear degenerate properties. A multi-layer perceptron (MLP) probe is better suited to capture these relationships than a simple linear hyperplane (logistic regression).

---

## Plan

### Phase 1: Biophysical Properties Mapping
- [x] Map all 20 amino acids to continuous biophysical scales inside `generate_probe_labels.py`:
  - Kyte-Doolittle Hydropathy Index
  - Residue Volume / Molecular Weight
  - Isoelectric Point (pI)
- [x] Extend `generate_probe_labels.py` to write these continuous targets to `probe_labels.csv`.

### Phase 2: Continuous Regression Probes
- [x] Implement a regression evaluation path in `probe_linear.py` (e.g. using Ridge Regression with cross-validation).
- [x] Train regression probes on the static codon embeddings to predict hydropathy, volume, and pI.
- [x] Report validation scores ($R^2$ and Pearson correlation).

### Phase 3: MLP Classifier Probes
- [x] Implement an MLP Classifier probe path in `probe_linear.py` using PyTorch or scikit-learn's `MLPClassifier`.
- [x] Evaluate MLP accuracy for 20-class AA identity and 4-class polarity.
- [x] Benchmark MLP probe performance against the baseline Logistic Regression probes.

### Phase 4: Comparative Reporting
- [x] Consolidate results into a report comparing:
  - Linear categorical accuracy (baseline)
  - MLP categorical accuracy (non-linear probe)
  - Regression $R^2$ scores on physical scales (continuous probe)
- [x] Document findings about what structural properties CodonLM learns implicitly.

---

## Done Definition

- Continuous biophysical values (hydropathy, volume, pI) are integrated into label generation.
- Ridge regression probes evaluate physical scales with cross-validated $R^2$ scores.
- MLP classifier probes evaluate categorical mappings and comparison reports are generated.
