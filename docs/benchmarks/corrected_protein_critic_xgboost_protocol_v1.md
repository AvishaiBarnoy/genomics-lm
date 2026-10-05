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

The fixed hyperparameter grid is selected on validation only: maximum tree
depth 3 or 6 and 200 or 500 estimators, with learning rate 0.05, row and column
subsampling 0.8, L2 regularization 1.0, one CPU worker, and a fixed seed.
Classification selection maximizes balanced accuracy; stability selection
minimizes MAE. After selection, XGBoost is refit on labelled train plus
validation examples, then evaluated on test once. Test comparisons report
balanced accuracy and macro-F1 for classification or MAE for stability, with
95% bootstrap intervals resampling protein clusters.

Example (after placing the exact best-checkpoint file locally and installing
the optional XGBoost dependency):

```sh
python scripts/benchmark_corrected_protein_critic_xgboost.py \
  --checkpoint runs/corrected-protein-critic-v1-seed1337/checkpoints/best_critic.pt \
  --out docs/benchmarks/corrected_protein_critic_xgboost_v1.json
```

This script does not establish that the critic is suitable for generation
guidance by itself. In particular, better XGBoost performance on critic
embeddings than on matched raw features is representation evidence, not proof
of generation benefit; the scaffold/stability sample-size limitations and
separate guidance evaluation remain relevant.
