# Corrected ProteinCritic XGBoost controls

`scripts/benchmark_corrected_protein_critic_xgboost.py` implements the next
corrected-critic exit-gate comparison. It requires the corrected-v2 manifest,
all three frozen JSONL splits, task vocabularies, the report-selected best
checkpoint, and the optional `xgboost` package. It refuses a checkpoint unless
its SHA-256 matches the `best` checkpoint recorded in
`corrected_protein_critic_training_v1.json`; a `last` checkpoint is not a
substitute.

For Pfam and EC, it compares the frozen critic with XGBoost trained on fixed raw
sequence features and with XGBoost trained on frozen critic bottleneck
embeddings. The raw features are log sequence length, amino-acid composition,
and normalized dipeptide/tripeptide frequencies. Stability also receives mean
Kyte-Doolittle hydropathy, net charge per residue, aromatic fraction,
aliphatic fraction, and cysteine fraction. Raw features and critic inputs use
the same maximum sequence window (`block_size - 2` residues); labels and
features are never used to build the split.

The hydropathy feature uses the Kyte-Doolittle scale ([Kyte & Doolittle,
1982](https://doi.org/10.1016/0022-2836(82)90515-0); [ExPASy ProtScale
table](https://web.expasy.org/protscale/pscale/Hphob.Doolittle.html)). It is a
sequence hydropathicity index, not a measured folding free energy or stability
value. The charge descriptor is deliberately approximate: K/R contribute +1,
D/E contribute -1, and other residues contribute zero; it is not pH-dependent
and omits termini and histidine.

The fixed hyperparameter grid is selected on validation only: maximum tree
depth 3 or 6 and 200 or 500 estimators, with learning rate 0.05, row and column
subsampling 0.8, L2 regularization 1.0, one CPU worker, and a fixed seed.
Classification selection maximizes balanced accuracy; stability selection
minimizes MAE. The selected estimator remains fitted on labelled training
examples only and is evaluated on test once. This intentionally matches the
ProteinCritic's training-data boundary; validation labels select hyperparameters
but are not added to XGBoost's fit set. A larger-data/scaling comparison should be
a separately named experiment with a common expanded training set for both models.
Test comparisons report
balanced accuracy and macro-F1 for classification or MAE for stability, with
95% bootstrap intervals resampling protein clusters.

For Pfam and EC, null task labels are excluded from that task's fit and scoring;
they are not mapped to an `unknown` class. These are closed-set classifiers over
the frozen eligible vocabularies (43 Pfam first-domain labels and seven top-level
EC labels). They cannot reject an unseen family/function without a separately
designed open-set calibration and evaluation.

An interval is reported only when at least two independent labelled test
clusters are available. In the current corrected-v2 split, stability has 62
labelled test proteins but just one cluster, so its cluster-bootstrap and
paired-improvement intervals are explicitly marked not estimable. The point
metrics remain descriptive for that held-out cluster and do not establish
generalization to new stability clusters.

The critic inference device defaults to `auto` (CUDA, then MPS, then CPU) and
can be set explicitly with `--device`. XGBoost remains CPU-based for this
benchmark.

Example (after placing the exact best-checkpoint file locally and installing
the optional XGBoost dependency):

```sh
python scripts/benchmark_corrected_protein_critic_xgboost.py \
  --checkpoint runs/corrected-protein-critic-v1-b2e32-seed1337/checkpoints/best_critic.pt \
  --out docs/benchmarks/corrected_protein_critic_xgboost_v1.json
```

This script does not establish that the critic is suitable for generation
guidance by itself. In particular, better XGBoost performance on critic
embeddings than on matched raw features is representation evidence, not proof
of generation benefit; the scaffold/stability sample-size limitations and
separate guidance evaluation remain relevant.
