# Directed ProteinEBM Corruption Plan

## Phase 0: Freeze The Baseline

- [ ] Freeze the current uniform-20% implementation, dataset manifest, critic
  checkpoint, EBM architecture, seed set, and compute budget.
- [ ] Add strict standard-amino-acid validation at dataset/trainer boundaries.
- [ ] Save per-decoy mutation provenance and establish uniform-only validation
  metrics with confidence intervals.

Exit gate: the present objective is exactly reproducible and invalid sequence
symbols cannot silently become unknown tokens.

## Phase 1: Corruption Strategy Contract

- [ ] Define a registered corruption-strategy interface with a supplied RNG and a
  structured decoy/provenance result.
- [ ] Validate mixture weights, mutation rates/counts, and strategy-specific options.
- [ ] Guarantee actual residue changes and exact interrupted/resumed reproduction.
- [ ] Add unit/property tests for alphabet, length, mutation count, provenance, and
  deterministic sampling.

Exit gate: adding a strategy does not require editing the EBM task or engine.

## Phase 2: Directed Negative Families

- [ ] Implement BLOSUM-conservative hard negatives.
- [ ] Implement Grantham-radical negatives with declared charge, polarity, and
  volume severity bands.
- [ ] Implement ProteinLM-low-likelihood contextual negatives without using EBM
  validation/test outcomes to choose candidates.
- [ ] Retain uniform substitutions as an independently selectable component.

Exit gate: each family passes provenance and distribution audits and produces a
nontrivial range of difficulty.

## Phase 3: Controlled Ablation

- [ ] Train uniform-only and each single-family strategy under matched conditions.
- [ ] Train predeclared candidate mixtures, including the initial equal 25% mixture.
- [ ] Report per-family ranking accuracy, energy gap, AUROC, calibration, and simple
  composition/charge/hydrophobicity baselines.
- [ ] Select one mixture using validation results and record the decision before
  opening test results.

Exit gate: the selected mixture improves hard-negative discrimination and is not
explained by a trivial physicochemical shortcut.

## Phase 4: External Validation And Promotion

- [ ] Evaluate energy ordering on an external experimental mutation-effect or
  stability dataset with family/scaffold-aware splits.
- [ ] Measure whether EBM ranking improves generated-sequence selection over the
  frozen ProteinCritic and ProteinLM likelihood controls.
- [ ] Document supported and unsupported interpretations of the energy score.
- [ ] Promote the mixture only if predefined validation gates pass; otherwise retain
  uniform-only as a diagnostic baseline and do not use EBM energy as a quality claim.

Exit gate: any promoted EBM has controlled synthetic and external evidence.
