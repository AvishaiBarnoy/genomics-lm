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

Hyperparameters are selected on validation only, and the selected estimator
remains fitted on labelled training examples only. The original broader grid
proved too costly for this CPU-only run, so the completed comparison uses the
two explicitly recorded candidates in the result artifact: (maximum depth 2,
50 estimators) and (maximum depth 3, 100 estimators). Both use learning rate
0.05, row and column subsampling 0.8, L2 regularization 1.0, one CPU worker, and
seed 1337. Classification selection maximizes balanced accuracy; stability
selection minimizes MAE. This matches ProteinCritic's training-data boundary;
validation labels select hyperparameters but are not added to XGBoost's fit set.
A larger-data/scaling comparison should be a separately named experiment with a
common expanded training set for both models. Test comparisons report balanced
accuracy and macro-F1 for classification or MAE for stability, with 95%
bootstrap intervals resampling protein clusters.

The raw-feature path now constructs n-gram frequencies with vectorized residue
codes and passes the resulting matrix to XGBoost as CSR sparse data. Cluster row
indices are cached before bootstrap resampling, avoiding repeated full-test-set
scans. These are runtime improvements; the fixed feature definitions and
train/validation/test membership are unchanged.

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

## Completed comparison

The reproducible machine-readable output is
[`corrected_protein_critic_xgboost_v1.json`](corrected_protein_critic_xgboost_v1.json).
It records the XGBoost version, checkpoint and dataset hashes, selected
hyperparameters, split counts, metrics, and cluster-bootstrap intervals.
Classification results below are balanced accuracy; positive paired improvement
means XGBoost is better than the frozen critic. Intervals are 95% protein-cluster
bootstrap intervals.

| Task | Frozen ProteinCritic | XGBoost, raw sequence | XGBoost, critic embeddings |
|---|---:|---:|---:|
| PFAM (473 labelled test proteins; 442 clusters) | 0.3018 [0.2615, 0.3410] | 0.4716 [0.4234, 0.5161]; improvement +0.1698 [0.1182, 0.2241] | 0.3781 [0.3281, 0.4234]; improvement +0.0763 [0.0351, 0.1198] |
| EC (517 labelled test proteins; 472 clusters) | 0.2380 [0.1883, 0.2898] | 0.2113 [0.1781, 0.2523]; improvement -0.0268 [-0.0881, 0.0310] | 0.2603 [0.2131, 0.3100]; improvement +0.0223 [-0.0172, 0.0568] |
| Stability (62 labelled test proteins; 1 cluster) | MAE 0.7618 | MAE 0.6953; improvement +0.0665 (interval not estimable) | MAE 0.9776; improvement -0.2158 (interval not estimable) |

PFAM is the clearest result: both XGBoost controls exceed the critic, and their
paired improvement intervals are positive. EC performance is statistically
inconclusive at this test-set size because both paired intervals include zero.
Stability results are descriptive only: all labelled test examples belong to a
single cluster, so no cluster-based uncertainty estimate is possible. These
results do not yet show that critic embeddings improve generation guidance; that
requires a separate controlled generation evaluation.

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
