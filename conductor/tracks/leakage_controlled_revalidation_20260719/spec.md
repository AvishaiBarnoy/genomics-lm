# Specification: Leakage-Controlled CodonLM Revalidation

## Status
Planned.

## Objective
Produce the first CodonLM result set trained and evaluated entirely under
leakage-controlled, provenance-complete protocols. Legacy Stage 2.6 artifacts remain
available for historical and engineering comparisons, but they cannot establish
corrected held-out performance.

This track is the execution plan for GitHub issue
[#92](https://github.com/AvishaiBarnoy/genomics-lm/issues/92).

## Why Retraining Is Deferred
Issues in tokenization, fragmentation, long-gene chunking, packing, leakage audits,
gradient accumulation, attention dropout, and vocabulary resolution all change the
training stream or training semantics. Retraining after each fix would spend compute
on artifacts that are immediately superseded. Each engineering PR therefore uses
unit fixtures and short smoke runs; full training begins only after the pipeline
freeze gate passes.

## Artifact Generations

### Legacy generation
- Existing Stage 2.6 datasets, checkpoints, embeddings, probes, and benchmark tables.
- May be used for historical comparison, debugging, and explicitly labeled transfer
  experiments.
- Must remain labeled `legacy protocol` and must not be promoted as corrected
  genome- or genus-held-out evidence.

### Corrected generation
- Source records are globally split before fragmentation, chunking, or packing.
- Dataset manifests capture source checksums, accession identities, split groups,
  tokenizer and packing policies, seeds, vocabulary hash, software versions, and
  audit results.
- Models are trained from random initialization after the pipeline freeze.
- Derived embeddings and probe results identify their dataset, checkpoint,
  vocabulary, code SHA, and evaluation split artifacts.

## Blocking Engineering Gates
- [x] #79: mandatory global genome-aware splitting.
- [x] #87: legacy scientific claims relabeled.
- [x] #91: required core CI, fatal lint, coverage, and checkout-cleanliness checks.
- [x] #80: ambiguous-codon boundary preservation and fragment provenance.
- [ ] #78: lossless long-gene chunking and explicit packing boundaries.
- [ ] #77: preventive exact-duplicate and protein-homology leakage audits.
- [ ] #83: abort and clear non-finite gradient-accumulation groups.
- [ ] #81: correct attention-dropout behavior in all attention paths.
- [ ] #84: tokenizer artifact as the vocabulary source of truth.
- [ ] #86: causal embedding extraction provenance and unsafe-fallback removal.
- [ ] #82: format-aware, vocabulary-safe perplexity baselines.
- [ ] #88: grouped DNA-shape controls and local-sequence baselines.
- [ ] #89: protein-cluster-held-out AMR evaluation and robust class reporting.

Issues #85 and #90 are useful follow-ups but do not block the first corrected
intrinsic and downstream result set.

## Pipeline Freeze Gate
Full retraining is prohibited until all blocking engineering issues are merged and
the following artifacts pass a reproducible preflight command:

- immutable source inventory and checksums;
- genome-held-out and genus-held-out split assignments;
- fragment and chunk provenance with source-record membership preserved;
- tokenizer vocabulary and policy hashes;
- packing configuration and deterministic seed;
- exact-duplicate and protein-cluster leakage reports with zero blocking violations;
- dataset token/count summaries and achieved split fractions;
- successful CPU tests plus MPS smoke training, checkpoint, and resume tests.

Any change to a frozen source, tokenizer, split, audit threshold, vocabulary, or
packing policy creates a new dataset version and invalidates dependent training runs.

## Retraining Protocol
- Train corrected models from random initialization; do not initialize from a model
  exposed to legacy holdout records.
- Run genome-held-out training for at least seeds `1337` and `2027`.
- Keep architecture, objectives, tokenizer, total non-PAD token exposure, and
  evaluation commands fixed across seeds.
- Report genus-held-out results separately; do not pool them with genome holdouts.
- Use the MPS runtime configuration only after its equal-token quality gate passes.
- Preserve configs, manifests, logs, checkpoints, consumed-token counters, wall time,
  and peak-memory metrics for every run.

## Corrected Evaluation Protocol
1. Compare uniform, unigram, bigram, trigram, and CodonLM loss, perplexity, and
   bits/codon on identical token streams.
2. Extract embeddings causally from corrected checkpoints with complete provenance.
3. Rerun EC, essentiality, AMR, and DNA-shape evaluations with controlled group or
   protein-homology holdouts and shared folds for all controls.
4. Report balanced accuracy, macro-F1, macro-AUPRC, AUROC where defined, and
   class-aware confidence intervals.
5. Audit generated sequences against training nucleotide and protein records before
   making novelty or memorization claims.

## Promotion Criteria
Corrected results may replace legacy headlines only when:

- every blocking issue and pipeline-freeze check passes;
- two genome-held-out seeds complete without invalid updates or audit violations;
- simple baselines and CodonLM use the same controlled test stream;
- downstream results use corrected checkpoints and embeddings;
- a versioned report contains commands, hashes, per-seed results, aggregate results,
  limitations, and failed gates;
- legacy and corrected protocols remain visibly separated.

## Relationship to Other Tracks
The [optimized training validation track](../optimized_training_validation_20260717/)
may establish runtime and equal-token engineering behavior using legacy artifacts.
Those results do not satisfy this track's scientific promotion criteria. Context
ablation intended for corrected claims must use the frozen corrected datasets and
the evaluation protocol defined here.

## Non-Goals
- Retrofitting corrected claims onto legacy checkpoints or embeddings.
- Changing model size or adding objectives during the controlled revalidation.
- Treating smoke runs, validation subsets, or CPU execution as final MPS evidence.
- Requiring generation-protocol or memmap optimizations (#85 and #90) before the
  first corrected intrinsic and probe report.
