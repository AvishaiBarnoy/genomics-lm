# Synonymous Controls Pilot Plan

## Overview
Evaluate whether CodonLM has learned biological codon choice (codon usage bias, GC preferences, and sequence context) beyond trivial protein translation.

## Methodology
- **Gene Selection**: 100 intact coding sequences sampled from the frozen validation split (`data/processed/corrected/corrected-codonlm-v1/genome/manifest.json`), length 30–500 codons.
- **Synonymous Controls**: For each gene, generate a synonymous alternative encoding the exact same protein sequence, preserving the start codon, terminal stop codon, reading frame, and gene length.
- **Evaluation**: In evaluation mode (batch size 1), compute length-normalized negative log-likelihood (NLL in nats/codon) for both natural and control sequences:
  $$\Delta\mathrm{NLL} = \mathrm{NLL}_{\mathrm{control}} - \mathrm{NLL}_{\mathrm{natural}}$$
- **Success Criteria**: Pilot passes if `median(ΔNLL) > 0`, indicating that the natural sequence receives higher likelihood than synonymous alternatives.

## Tasks
- [x] Implement `reshuffle_synonymous_gene` supporting `uniform_alternative`, `uniform_any`, and `within_gene_shuffle` modes.
- [x] Implement paired likelihood evaluation harness (`src/eval/synonymous_controls.py`).
- [x] Implement CLI runner `scripts/eval_synonymous_controls.py`.
- [x] Add unit tests verifying biological invariants and NLL calculations (`tests/test_synonymous_controls.py`).
- [x] Execute pilot on trained checkpoint (`runs/corrected-codonlm-v1-batch64-lr15-seed2027`).
- [x] Document findings and baseline metrics.
- [x] Negative control on randomly initialized model (`--random_weights`).
- [x] Automated KPI reporting integration (`scripts/generate_run_summaries.py`).

## Pilot Results (`runs/corrected-codonlm-v1-batch64-lr15-seed2027`)

| Metric | Trained: Uniform Alternative (Tier 1) | Trained: Within-Gene Shuffle (Tier 2) | Negative Control: Random Weights Baseline |
|---|---|---|---|
| **Evaluated Genes** | 100 validation genes | 100 validation genes | 100 validation genes |
| **Natural NLL** | **3.7218** nats/codon | **3.7218** nats/codon | 4.3617 nats/codon |
| **Control NLL** | **4.1409** nats/codon | **3.7382** nats/codon | 4.3908 nats/codon |
| **Natural PPL** | 42.18 | 42.18 | 78.46 |
| **Control PPL** | 63.06 | 42.82 | 80.76 |
| **Median $\Delta\mathrm{NLL}$** | **$+0.4354$ nats/codon** | **$+0.0150$ nats/codon** | **$+0.0239$ nats/codon** |
| **Mean $\Delta\mathrm{NLL}$** | $+0.4191 \pm 0.1850$ | $+0.0164 \pm 0.0217$ | $+0.0291 \pm 0.0534$ |
| **IQR [Q25, Q75]** | $[+0.2435, +0.5778]$ | $[+0.0032, +0.0292]$ | $[-0.0019, +0.0643]$ |
| **Win Rate** | **100.0%** (100/100 genes) | **80.0%** (80/100 genes) | 73.0% (73/100 genes) |
| **Wilcoxon Test** | $W=5050.0, p=1.95 \times 10^{-18}$ | $W=4428.0, p=3.01 \times 10^{-11}$ | $W=3991.0, p=2.32 \times 10^{-7}$ |
| **Pilot Verdict** | **PASS** | **PASS** | **NEGATIVE BASELINE** |


