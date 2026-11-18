# Corrected CodonLM Training Program Plan

## Status

In progress. The locked basic genome-held-out CodonLM now clears the trigram
baseline on the frozen test set in two seeds (PPL `39.133` and `39.492` versus
`42.037`); the previous seed-1337 result and failed pilot/ablation results below
are historical, not the current primary result. The separate genus-held-out primary
run has not been trained. The corrected ProteinCritic's report-selected checkpoint
is present and hash-verified locally. The matched XGBoost comparison completed
and was merged with its versioned metrics and protocol in
`docs/benchmarks/corrected_protein_critic_xgboost_v1.json` and
`docs/benchmarks/corrected_protein_critic_xgboost_protocol_v1.md` (PR #184).
The optional XGBoost package was used from a temporary environment; the project
Conda environment remains unchanged. Generation termination/replay and biological
representation claims remain separately gated as described below.

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
- [x] Train and evaluate the identical genome-held-out primary model at seed `2027`.
- [ ] Train the separately labelled genus-held-out primary model from random
  initialization.
- [ ] Verify matched architecture, objective, non-PAD exposure, and config identity
  across comparable runs.
- [ ] Archive complete run manifests, checkpoints, logs, and failure telemetry.

Exit gate: all primary runs finish without leakage, OOM, invalid update, counter
mismatch, or provenance failure.

The genus manifest assigns `Helicobacter` and `Streptomyces` to test and
`Bacillus` and `Borreliella` to validation; those genera are excluded from the
genus-run training split. The run itself remains pending, so these are split
assignments, not completed generalization results.

## Phase 3: Evaluate and Decide on the Primary Model

The initial seed-1337 intrinsic result, recorded in
`docs/CORRECTED_PRIMARY_INTRINSIC_EVALUATION.md`, failed the trigram gate. It was
superseded by the matched batch-64/LR-`1.5e-4` primary condition below, which clears
the trigram gate on frozen test data for both seeds. Earlier diagnostics and
regularization/effective-batch experiments below are retained as history, not as
the current primary result. The separate genus-held-out run remains outstanding.

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
- [ ] If a later claim needs tighter uncertainty across training runs, predeclare
  one additional seed of the locked batch-64, LR `1.5e-4` basic configuration.
  The existing 1337/2027 replication passes the trigram gate; this is deferred,
  not a prerequisite for the current evaluation and inference work.
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
  The replicated 50-prompt generation gate is complete for both corrected seeds:
  raw sampling has 0% natural stops and 100% hard-cap failures, while the
  target-length CDS constraint also has 0% natural stops. Generated GC rises to
  74-76% versus 52.9% in held-out sources. Exact indexed 10/20-codon coverage is
  zero. The exhaustive nucleotide/protein audit now completes with minimap2 plus
  bounded MMseqs2 target batches. Stop-probability diagnostics show termination
  ranks near 61 at natural gene ends; top-k 5/20 removes termination from the
  sampling support. A 10-prompt pilot restores 90% natural stopping for both seeds
  at temperature 1.0 without top-k truncation. Run the larger corrected decoder
  baseline before promoting termination/replay training.
  The 50-prompt confirmation reached only 70% and 56% natural stops across seeds,
  with 30% and 42% hard-cap rates. Completed samples were generally shorter than
  held-out CDSs. The unrestricted decoder is retained as the base comparison, but
  the confirmation now authorizes the Phase 6 termination-head ablation.
- [ ] Publish per-seed and aggregate primary results with confidence intervals and
  limitations, then record a go/no-go decision for extensions.

Exit gate: the basic model outperforms the best simple intrinsic baseline and the
corrected report passes its promotion criteria. Otherwise pause and audit.

## Phase 4: Revalidate the External ProteinCritic

- [x] Select and freeze one critic architecture rather than mixing historical
  average-pooled, structural-transfer, and bidirectional variants.
- [x] Freeze protein sources, label definitions, task vocabularies, preprocessing,
  and train/validation/test artifacts with exact provenance.
- [x] Split translated proteins by sequence-homology clusters and report label/class
  balance, missing classes, cluster thresholds, and cross-split nearest neighbors.
  Corrected critic v1 uses a from-scratch bidirectional 8L8H-d256 backbone with
  attention pooling and no motif-saliency regularizer or legacy checkpoint transfer.
  MMseqs2 clustering at 30% identity and 80% coverage produced 14,689 source
  clusters. After removing records without a retained target, v2 contains 15,054
  records in 5,149 disjoint retained clusters. Supported targets are 43 first-domain Pfam classes,
  seven top-level EC classes, and continuous MegaScale `deltaG`. Stability has eight
  train, one validation, and one test scaffold cluster, so it is a limited
  scaffold-held-out regression rather than a universal stability probability.
- [x] Retrain Pfam-family, EC-function, stability, and declared structural/protein-
  type heads under the corrected split; do not initialize from a holdout-exposed
  critic unless the transfer protocol proves compatibility and isolation.
  The regression-aware, manifest-bound trainer and evaluator are implemented. The
  corrected 8L8H-d256 run completed in 139 minutes with batch 2, accumulation 16,
  and context 512. Epoch 9 is the validation-selected checkpoint. Pfam/EC learn
  nontrivial held-out signal, while stability generalization varies by scaffold;
  the eight training stability clusters contain 930/47/46/29/29/21/13/7 records.
- [ ] Calibrate every probability-producing head and report class-aware metrics,
  confidence intervals, reliability, and generated-protein OOD behavior.
  Before further test evaluation, run one validation-selected class-balance ablation
  against the completed seed-1337 baseline. The only experimental change is
  training-split square-root inverse-frequency weighting (maximum 4x) for Pfam and
  EC cross-entropy; architecture, data, seed, context, batch/accumulation, learning
  rate, and ten-epoch budget remain fixed. Validation loss stays unweighted. Promote
  only if both heads improve balanced accuracy or macro-F1 without more than a
  three-point absolute top-1 loss, and stability validation MAE regresses by no more
  than 5%. Do not inspect test metrics until that validation decision is recorded.
  The first attempt stopped progressing at epoch 1 microbatch 4,400 before its
  first 30-minute checkpoint. The observed macOS wait state is consistent with
  either system sleep (for example, lid closure) or an MPS driver wait; the logs
  cannot distinguish them. It produced no epoch or checkpoint artifact and is
  excluded from evaluation. Retry 1 changes only the run ID and periodic checkpoint
  cadence (five minutes) so a future interruption loses at most a bounded interval.
  Retry 1 ultimately completed after recovery. Validation-only evaluation rejects
  class weighting: Pfam balanced accuracy/macro-F1 fell from 0.3090/0.2607 to
  0.2846/0.2484, although EC improved from 0.2190/0.2093 to 0.2419/0.2288 and
  stability MAE improved from 1.0924 to 1.0486. The test split remains sealed.
- [x] Compare the corrected critic with XGBoost on the identical frozen splits.
  Pfam and EC controls use training-fitted amino-acid composition, sequence length,
  and dipeptide/3-mer features; stability additionally uses declared
  physicochemical descriptors. Select XGBoost hyperparameters on validation only
  and report the same class-aware or regression metrics and confidence intervals.
- [x] Run separate XGBoost probes on frozen ProteinCritic embeddings and raw
  sequence features. Treat improved embedding-probe performance as representation
  evidence only when it also exceeds the matched raw-feature XGBoost control.
- The completed comparison is documented in PR #184 and its versioned report.
  XGBoost exceeded the frozen critic on PFAM using both raw sequence features
  (balanced accuracy 0.4716 vs 0.3018; paired 95% cluster-bootstrap improvement
  interval 0.1182–0.2241) and critic embeddings (0.3781; interval 0.0351–0.1198).
  EC differences were inconclusive because the paired intervals included zero.
  Stability results remain descriptive only: the 62 labelled test examples came
  from one protein cluster, so a cluster-bootstrap interval was not estimable.
  Keep this v2 split frozen and do not replace the selected best checkpoint with
  `last`.
- A post-hoc class-level decomposition is recorded in
  `docs/benchmarks/corrected_protein_critic_xgboost_class_diagnostics_v1.md` and
  its JSON. Raw XGBoost recall exceeded the critic in 25/43 PFAM classes; the
  critic was higher in 11 and tied in 7. EC differences vary by class, with the
  raw model more concentrated on class 2. Most PFAM classes have fewer than 20
  test examples, so these per-class test observations are descriptive and must
  not be used to tune a follow-up model.
- [ ] Use validation-only per-class support and error analysis to decide whether
  a focused PFAM critic-head or loss ablation is justified. Keep the v2 test
  report frozen; use a fresh external/frozen test set for confirmatory claims.
- [ ] Version the passing critic checkpoint and bind it to its dataset, labels,
  architecture, and calibration artifacts.

### Stability scaffold-diversity follow-up (not a replacement for the v2 benchmark)

- [x] Keep the current v2 XGBoost-vs-Critic benchmark on its frozen split as the
  first controlled algorithm comparison. Do not silently add records to that split.
- [ ] Audit additional experimental stability sources for raw measurement
  semantics, protein/domain identifiers, assay conditions, and reuse permissions.
  The source release to inspect includes the [Mega-scale experimental analysis
  data tables](https://zenodo.org/records/7992926).
  The current `dG_extdG_data_Fig1.csv` source contributes only 10 distinct PDB IDs
  to the present stability task. A separate local Fig. 5 table has 104 PDB IDs,
  but contains derived per-site substitution quantities rather than the same
  protein-level `deltaG` target; do not treat it as interchangeable labels without
  validating its exact target semantics.
- [ ] Build a separately versioned stability dataset with broader parent-domain
  coverage. Keep all variants from a parent domain together and cluster related
  domains before splitting. Freeze multiple independent validation and test
  scaffold groups before training.
- [ ] Predeclare whether the endpoint is absolute folding `deltaG`, mutation
  `delta-deltaG`, or a thresholded assay-specific foldability label. Do not mix
  endpoints or equate structure-prediction confidence with experimental stability.
- [ ] Train critic and XGBoost controls on the same expanded training examples and
  evaluate the same held-out scaffold groups. Size the number of held-out groups
  using a power/precision analysis; a large number of variants from one scaffold
  does not substitute for scaffold diversity.

Exit gate: the corrected critic is suitable for its declared ranking or calibrated-
probability use. Until then, legacy critic outputs are exploratory only and cannot
support promotion, stability, family, function, or guidance claims.

## Phase 5: Multi-Offset `n+x` Ablation

- [x] Predeclare offsets (including whether `n+2` is used), weights, projection-head
  initialization, backbone-freeze/joint-training policy, token budget, and metrics.
- [x] State separately whether offset logits are auxiliary training signals or are
  consumed by a merged-prior decoder at inference.
  The first corrected condition uses independent two-layer projection heads whose
  linear matrices are identity-initialized but whose intervening GELU means the
  complete projection is not initially an identity function. It evaluates
  projection heads at `+2/+4/+8/+16/+32`, equal weights of `0.1`, three frozen-
  backbone epochs, effective batch 64, and adaptive 10% warmup. These offsets are
  future-token distances, not direct labels for helices, sheets, or contacts. The
  main next-token head remains frozen. Prior merging is disabled during training
  and is evaluated later as a separately labelled decoder condition against raw
  next-token sampling.
- [x] Train the predeclared head-only multi-offset condition from the corrected primary checkpoint without
  changing data splits or the main next-token head.
- [x] Report main next-token loss and every offset loss separately; rerun long-range,
  downstream, termination, runtime, and memory evaluations.
- [x] Reject merged-prior decoding at the tested weights using matched seeds.
  The run completed three epochs cleanly. All 176 shared anchor tensors remained
  bitwise identical, so ordinary logits, hidden-state probes, and downstream
  embeddings are unchanged by construction. Frozen-test next-token NLL/PPL remained
  `3.66696`/`39.13`. Offset heads improved NLL only `0.98-1.12%` over the
  unprojected next-token head, with no decay from `+2` to `+32`. Across 160 matched
  generations, equal-weight prior merging reduced natural stops from `30.6%` to
  `6.25%` and increased hard caps from `69.4%` to `93.8%`; it lost 39 control stops,
  preserved 10, and rescued none. Retain the heads only as exploratory probes and
  do not promote this merged-prior decoder. A second training seed is unnecessary
  for rejection but would be required before making a positive representation claim.

Exit gate result: failed. Next-token quality was preserved, but the weak,
distance-flat probe gain did not establish long-range structure and prior-guided
decoding materially degraded termination.

## Phase 6: Termination and Replay Ablation

- [x] Predeclare distance buckets, replay construction, matched prompts/seeds,
  decoding conditions, token budget, and acceptance thresholds.
  The head-only condition uses EOS-distance buckets `[0,3,10,30]`, square-root
  inverse-frequency weights measured on 25,238,438 frozen training positions, one
  epoch from the promoted seed-1337 checkpoint, joint LR `5e-6`/head LR `1e-4`,
  and no replay. Raw unrestricted temperature-1.0 decoding is primary;
  head-biased decoding is a separately labelled intervention. Promotion permits
  at most 2% test-NLL regression and requires fewer hard caps without short-length
  collapse.
- [x] Train the termination-head condition without replay from the corrected primary
  checkpoint. The frozen test NLL regression is 1.10%, within the 2% gate, but
  raw unrestricted termination remains 63/100, identical to the anchor aggregate.
  The head predicts only exact-boundary and far classes (balanced accuracy 36.44%;
  zero recall for all three intermediate buckets), so this condition is not promoted.
- [x] Add generated-prefix replay only if the head-only condition is insufficient.
  Head-only is insufficient; the replay condition is authorized and frozen. Replay
  uses 79 unrestricted hard-cap failures generated from 200 training-split prefixes,
  exact `[0,3,10,30]` tail labels, one replay batch per optimizer group, and no
  validation/test source records. The one-epoch condition is complete.
- [x] Compare natural, syntax-constrained, replay-trained, and decoder-biased
  behavior without conflating training and inference interventions. Replay changes
  raw behavior intrinsically; strict class-0 decoder bias applies zero steps and is
  reported separately.
- [x] Report length distributions, natural-stop, early-stop, and hard-cap rates plus
  primary loss, sequence controls, runtime, and memorization. Replay reduces hard
  caps from 37/100 to 19/100 (`p=7.6e-6`) with a 1.31% primary-test NLL regression
  and no measured training-sequence overlap. Median generated length falls from 207
  to 147.5 codons, so the result requires an independent training replicate before
  promotion over the corrected primary.

Exit gate: natural completion improves without forced-stop dependence,
short-sequence collapse, or material primary-quality regression.

Screening status: passed, but not promoted. Natural completion improves within the
NLL gate and without forced stopping; the material length shift and poorly calibrated
auxiliary classes require independent replay replication.

### Follow-on: Natural-Termination Ablation Matrix (proposed; not yet run)

Terminology guard: in the corrected raw-generation protocol, a hard cap is a maximum
generation length. It truncates an unfinished sequence and is counted as a failure;
it does **not** append a stop codon. The separate `require_terminal_stop` mode keeps
sampling until a stop token is generated or the cap is reached. A decoder-forced
terminal marker, if ever tested, must be reported as forced completion, never as a
natural stop.

Use a staged rather than full-factorial matrix, with identical frozen prompts,
sample seeds, model checkpoint, and length-matched evaluation:

- [ ] **Decode-only controls:** unrestricted raw decoding with full vocabulary and
  no forced stop; a predeclared temperature/top-k sweep that keeps stop codons in
  the sampling support; and a calibrated stop-logit/hazard intervention, optionally
  gated by a validation-selected minimum length. Keep syntax/length-constrained or
  forced-terminal output as engineering controls, not intrinsic model results.
- [ ] **Token-loss intervention:** compare the standard objective with a small,
  validation-selected terminal-stop-token loss-weight sweep. Track premature stops
  as a co-primary guardrail so a higher stop rate cannot win by collapsing lengths.
- [x] **Distance-to-stop auxiliary head:** screened once; intermediate distance
  buckets were poorly calibrated and it did not improve natural stopping alone.
- [ ] **Generated-state replay:** replicate the current hard-cap replay condition
  with an independent training seed; retain the original seed as the anchor.
- [ ] **Length-conditioned option:** only if the decode/loss/replay comparisons do
  not meet the gate, prototype explicit target-length or stop-hazard conditioning
  as a separately labelled training objective.

Every candidate reports natural-stop, premature-stop, EOS, hard-cap and forced-stop
rates separately, plus generated-length distributions, frozen-test NLL/PPL, and
uncertainty. Compare termination and length distributions with held-out complete
CDSs; do not select on test. Any forced completion remains outside the natural-stop
acceptance metric. Freeze the matrix values and promotion thresholds before any
additional training.

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
