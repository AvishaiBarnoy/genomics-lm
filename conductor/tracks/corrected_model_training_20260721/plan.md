# Corrected CodonLM Training Program Plan

## Status

In progress, with the intrinsic gate paused for diagnosis. Dataset, evaluator,
generation-protocol, MPS runtime, immutable primary-config, and bounded MPS pilot
gates are complete. The genome seed-1337 primary run completed, but its unsmoothed
test PPL `48.267` did not beat the trigram baseline `42.037`, and natural sequence
was indistinguishable from the codon-order shuffle. Internal CodonLM extensions and
the external ProteinCritic remain gated and cannot conceal this failure.

## Phase 0: Freeze Primary Contracts

- [x] Freeze the 24-source genome/genus datasets, vocabulary, packing, leakage
  reports, evaluator contracts, and generation protocols.
- [x] Select the corrected MPS policy: batch 4, accumulation 32, checkpointing, AMP,
  MHA/SDPA, separator mask, and batch-aware mmap.
- [x] Add immutable genome-seed-1337, genome-seed-2027, and genus primary configs.
- [x] Pin the training-token budget, optimizer/scheduler, validation/checkpoint
  cadence, output naming, and pilot limits.
- [x] Add config-contract tests requiring random initialization and rejecting shape,
  offset, termination, replay, critic/energy, RoPE, SwiGLU, GQA, or other undeclared
  primary objectives and architectures.

Exit gate: no unresolved choice can alter the primary training stream, objective,
architecture, exposure, or provenance.

## Phase 1: Run the Bounded Primary Pilot

The first lifecycle run completed on 2026-07-23 with exact resume counters, zero
invalid groups, stable MPS memory, and validation loss `4.031`, but exposed a
compressed one-epoch cosine horizon and segment-only training-loss reporting. Those
results are diagnostic only. A schema-v2 segment then verified the 5,000-step
scheduler but found that pending, uncommitted microbatch losses entered checkpoint
metrics. Schema v3 then completed the full frozen epoch across six MPS invocations:
500 optimizer/scheduler steps, 25,238,438 committed tokens, exact metric boundaries,
zero invalid groups, validation loss `3.934` (PPL `51.10`), and stable 1.16 GB
allocated / 2.45 GB driver MPS memory. Evidence is recorded in
`docs/CORRECTED_PRIMARY_PILOT.md`.

- [x] Train from random initialization on a bounded portion of the frozen
  genome-held-out stream using the exact primary model and runtime policy.
- [x] Verify initial loss scale, finite gradients, committed non-PAD tokens,
  optimizer/scheduler counters, validation, and wall-time estimates.
- [x] Verify `last`/`best` checkpoint creation and exact resume without replaying or
  omitting committed updates.
- [x] Record peak host/MPS memory, throughput, non-finite groups, termination reason,
  and resolved provenance.
- [x] Approve the immutable configs for full training or revise them through a new
  versioned config contract and repeat the pilot.

Exit gate: pilot and resume complete on MPS without OOM, non-finite update,
counter/provenance mismatch, or an unexplained loss anomaly.

## Phase 2: Train the Primary Basic Model

- [x] Train the genome-held-out primary model at seed `1337`.
- [ ] Train the identical genome-held-out primary model at seed `2027`.
- [ ] Train the separately labelled genus-held-out primary model from random
  initialization.
- [ ] Verify matched architecture, objective, non-PAD exposure, and config identity
  across comparable runs.
- [ ] Archive complete run manifests, checkpoints, logs, and failure telemetry.

Exit gate: all primary runs finish without leakage, OOM, invalid update, counter
mismatch, or provenance failure.

## Phase 3: Evaluate and Decide on the Primary Model

Interim seed-1337 intrinsic evaluation is recorded in
`docs/CORRECTED_PRIMARY_INTRINSIC_EVALUATION.md`. It beats unigram but not bigram
or trigram and therefore fails the promotion gate. Pause dependent downstream and
generation evaluation. The mask audit passed; context ablation showed all useful
gain saturating at four input tokens, with no gain from longer context and a paired
`+0.13819` nats/token deficit to trigram. Run the predeclared regularization matrix
before considering an architecture extension. The checklist remains open until all
Phase 2 runs and matched evaluations are complete.

The four-condition regularization matrix is complete. At identical two-epoch
exposure, the untied/no-smoothing/dropout-0.05 variant reached validation PPL
`45.210`, compared with `49.167` for the reference. It remains behind the validation
bigram (`43.927`) and trigram (`42.459`), so the primary gate remains failed. Carry
the untied variant into a matched effective-batch-size ablation before considering
an architectural intervention.

- [x] Complete the matched regularization ablation and evaluate best checkpoints
  with manifest-bound unsmoothed validation NLL.
- [x] Run the token-matched effective-batch ablation. Reuse the completed
  batch-128 untied condition and train random-initialized batch-64 and batch-32
  conditions at 2,000 and 4,000 optimizer steps respectively. Keep physical batch,
  seed, data order, model, learning rate, two-epoch token exposure, and validation
  selection fixed.
- [x] Run a narrow effective-batch-64 learning-rate ablation. Compare peak rates
  `3e-4`, `2.25e-4`, and `1.5e-4` in fresh runs. Use scheduler-relative 10% warmup
  (200/2,000 steps), scale embedding and minimum rates with the backbone rate, and
  hold seed, token exposure, scheduler shape, and validation-only selection fixed.
- [x] Replicate the selected batch-64, LR `1.5e-4` condition with declared seed
  2027. Validation PPL is `40.961` for seed 1337 and `41.436` for seed 2027,
  below trigram `42.459` in both runs. The paired CodonLM-minus-trigram confidence
  intervals are entirely below zero. Lock this configuration for final frozen-test
  evaluation.
- [x] Run validation context ablation and a paired packed-window trigram comparison
  for the selected batch-64 checkpoint. Context gains continue through 32-128
  codons; the trigram deficit is `+0.015280` nats/token with 95% CI
  `[+0.014204, +0.016337]`.
- [x] Defer the conditional architecture intervention because the optimized basic
  model no longer trails trigram. If later work reopens it, predeclare a
  zero-initialized local
  causal-convolution or amino-acid/codon-factorization ablation based on
  transition-level errors. Preserve the demonstrated long-context gain and keep any
  explicit Markov-logit residual separately labelled as a hybrid model.
- [x] Evaluate unigram, bigram, trigram, and both locked CodonLM replicates on
  identical frozen-test tokens; report loss, perplexity, bits/codon, and improvement
  over the best baseline. Seed-1337 and seed-2027 PPL are `39.133` and `39.492`,
  both below trigram `42.037`.
- [x] Extract causal AMR embeddings for both corrected seeds with
  dataset/checkpoint/vocabulary/code provenance. Other downstream datasets remain
  pending.
- [ ] Run EC, essentiality, AMR, and DNA-shape evaluations with controlled splits and
  shared controls.
  EC preflight is currently blocked: all 6,617 matched legacy EC annotations occur
  in pretraining-train genomes and none in pretraining-test genomes. The controlled
  CARD AMR protein-cluster split passes its exact-pretraining-overlap gate after
  quarantine (3,733 train/1,285 test across six classes). Its report discloses 34
  protein clusters shared with pretraining.
  The first AMR representation gate fails: both random-Transformer controls
  outperform both pretrained final-layer causal-mean representations. Predeclare
  pooling/layer selection using grouped cross-validation within probe training
  before touching the AMR test set again.
  The train-only ablation selected layer-2 content mean (macro-AUPRC 0.4587 across
  grouped folds and seeds). Locked test balanced accuracy improved to 0.501/0.469
  from 0.322/0.349, but the representation still does not consistently beat both
  random controls; AMR-specific pretraining benefit remains unproven.
  The corrected linear DNA-shape gate also fails. Across two checkpoint seeds,
  final and layer-2 states are substantially worse than matched random-Transformer,
  one-hot, and local-sequence controls under both two-genome transfer and
  five-fold gene-grouped sensitivity protocols.
- [ ] Run raw and syntax-constrained generation with memorization and nucleotide/
  protein nearest-neighbor audits; do not use critic scores for promotion yet.
- [ ] Publish per-seed and aggregate primary results with confidence intervals and
  limitations, then record a go/no-go decision for extensions.

Exit gate: the basic model outperforms the best simple intrinsic baseline and the
corrected report passes its promotion criteria. Otherwise pause and audit.

## Phase 4: Revalidate the External ProteinCritic

- [ ] Select and freeze one critic architecture rather than mixing historical
  average-pooled, structural-transfer, and bidirectional variants.
- [ ] Freeze protein sources, label definitions, task vocabularies, preprocessing,
  and train/validation/test artifacts with exact provenance.
- [ ] Split translated proteins by sequence-homology clusters and report label/class
  balance, missing classes, cluster thresholds, and cross-split nearest neighbors.
- [ ] Retrain Pfam-family, EC-function, stability, and declared structural/protein-
  type heads under the corrected split; do not initialize from a holdout-exposed
  critic unless the transfer protocol proves compatibility and isolation.
- [ ] Calibrate every probability-producing head and report class-aware metrics,
  confidence intervals, reliability, and generated-protein OOD behavior.
- [ ] Version the passing critic checkpoint and bind it to its dataset, labels,
  architecture, and calibration artifacts.

Exit gate: the corrected critic is suitable for its declared ranking or calibrated-
probability use. Until then, legacy critic outputs are exploratory only and cannot
support promotion, stability, family, function, or guidance claims.

## Phase 5: Multi-Offset `n+x` Ablation

- [ ] Predeclare offsets (including whether `n+2` is used), weights, projection-head
  initialization, backbone-freeze/joint-training policy, token budget, and metrics.
- [ ] State separately whether offset logits are auxiliary training signals or are
  consumed by a merged-prior decoder at inference.
- [ ] Train matched multi-offset runs from the corrected primary checkpoint without
  changing data splits or the main next-token head.
- [ ] Report main next-token loss and every offset loss separately; rerun long-range,
  downstream, termination, runtime, and memory evaluations.
- [ ] Promote or reject the extension using replicated predeclared gates.

Exit gate: a replicated long-range/downstream gain does not materially degrade
next-token quality, termination, memory, or runtime reliability.

## Phase 6: Termination and Replay Ablation

- [ ] Predeclare distance buckets, replay construction, matched prompts/seeds,
  decoding conditions, token budget, and acceptance thresholds.
- [ ] Train the termination-head condition without replay from the corrected primary
  checkpoint.
- [ ] Add generated-prefix replay only if the head-only condition is insufficient.
- [ ] Compare natural, syntax-constrained, replay-trained, and decoder-biased
  behavior without conflating training and inference interventions.
- [ ] Report length distributions, natural-stop, early-stop, and hard-cap rates plus
  primary loss, sequence controls, runtime, and memorization.

Exit gate: natural completion improves without forced-stop dependence,
short-sequence collapse, or material primary-quality regression.

## Phase 7: Biophysical Shape-Guidance Ablation

- [ ] Freeze the shape-encoder artifact, targets, training sources, and relationship
  to every CodonLM and ProteinCritic heldout group.
- [ ] Train the corrected primary plus a frozen shape encoder.
- [ ] Train the corrected primary plus a jointly unfrozen encoder using recorded
  discriminative learning rates and matched token exposure.
- [ ] Run grouped DNA-shape evaluation with one-hot, random-model, 5-mer, and 7-mer
  controls on shared folds.
- [ ] Rerun synonymous and de novo generation with matched seeds, confidence
  intervals, absolute effects, and memorization audits. Use corrected critic scores
  only if Phase 4 passed.
- [ ] Promote or reject shape guidance independently of termination and multi-offset
  objectives.

Exit gate: replicated shape-guided improvement survives sequence controls and does
not depend on leakage, unmatched decoding, an invalid critic, or proxy-only claims.

## Phase 8: Combined Candidate and Generation Interventions

- [ ] Combine only independently promoted internal extensions and predeclare their
  initialization, objectives, weights, and rationale.
- [ ] Train a matched combined candidate; retain every independent ablation.
- [ ] Evaluate raw, syntax-constrained, decoder-biased, ReD, corrected-critic-guided,
  and any EBM-guided generation as distinct protocols with matched seeds and budgets.
- [ ] Treat ProteinCritic, EBM, ReD, and decoder constraints as external inference
  interventions unless a separately declared training objective explicitly uses one.
- [ ] Rerun intrinsic, downstream, generation, memorization, runtime, and provenance
  suites for the combined candidate.

Exit gate: each combined gain remains attributable, and no external intervention is
misreported as an intrinsic property of the generator.

## Phase 9: Publish the Corrected Program

- [ ] Publish a versioned comparison of the primary, each internal extension, the
  corrected ProteinCritic, the combined candidate, and external interventions.
- [ ] Report exact commands, hashes, seeds, token exposure, confidence intervals,
  absolute effects, failed gates, and limitations.
- [ ] Update repository headline tables while preserving a separate legacy section.
- [ ] Record final promotion/rejection decisions and create follow-up issues for
  unresolved failures.

Exit gate: corrected evidence supports every selected component, and each claimed
gain is traceable to a controlled experiment.

## Global Rules

- No legacy checkpoint initializes corrected primary training.
- Implemented code is not enabled unless its phase and immutable config declare it.
- No existing legacy checkpoint is treated as an all-extension model: the replay
  lineage has offset plus termination/replay components, while the shape-guided
  lineage has shape plus termination but no offset heads.
- ProteinCritic is external to CodonLM and does not block primary training, but its
  corrected gate blocks critic-based evaluation, promotion, and guidance claims.
- A failed primary or extension gate stops dependent work; later components cannot
  conceal the failure.
