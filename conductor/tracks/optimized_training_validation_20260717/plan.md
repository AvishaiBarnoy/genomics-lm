# Optimized Training Quality Validation & Context Ablation Plan

## Status
Planned.

## Purpose
Turn the July 2026 MPS throughput benchmark into a production training policy without
trading away model quality. First validate the measured runtime winner against the
current training behavior under equal data exposure. Only after that gate passes,
measure whether shorter contexts preserve held-out and downstream biological quality.

## Fixed Decisions
- Use `runs/2026-06-15_stage2.6_10L8H_d384_e10/checkpoints/best.pt` as the
  common starting checkpoint for the quality comparison.
- Use the existing Stage 2.6 train, validation, and test splits under
  `data/processed/stage2.6_large_master_pack/`; do not regenerate or reshuffle them
  during the runtime comparison.
- The reference is batch 4, gradient accumulation 32, activation checkpointing on,
  AMP on, standard multi-head attention, and bucket batching off.
- The candidate is batch 8, gradient accumulation 16, activation checkpointing off,
  AMP on, standard multi-head attention, and dynamic bucket batching with 8 buckets.
- Both variants retain an effective sequence batch of 128 and consume every training
  sequence once per epoch. Compare quality after the same number of non-PAD targets,
  not after the same number of optimizer steps or minutes.
- Run paired seeds `1337` and `2027`. Each pair must start from the same checkpoint
  and use the same data ordering seed.
- Keep historical and production configs unchanged until the acceptance gate passes.
- Do not promote batch 16, GQA, or FP32-at-batch-8: the MPS benchmark found them
  slower or memory-unstable on the target M2/8 GB machine.

## Acceptance Gates
The optimized candidate may become the recommended MPS configuration only if all of
the following hold across the paired runs:

- At least 1.5x lower mean end-to-end training wall time for equal non-PAD targets.
- Mean validation next-token loss is no more than 1% worse than the reference.
- Mean test perplexity is no more than 2% worse than the reference.
- Natural termination rate is no more than 5 percentage points below the reference.
- Median genomic quality/alignment score is no more than 5% below the reference.
- No new OOM failure, non-finite loss, skipped optimizer update, or resume failure.

If the candidate fails a quality gate, test the measured conservative fallback:
batch 4, accumulation 32, checkpointing off, AMP on, standard attention, and 8
buckets. Do not change production defaults unless either candidate passes every gate.

## Phase 1: Reproducible Paired Validation Harness
- [ ] Add immutable reference and candidate experiment configs named
  `configs/stage2.6_mps_quality_reference.yaml` and
  `configs/stage2.6_mps_quality_candidate.yaml`.
- [ ] Add config-validation tests proving both experiments use the same checkpoint,
  architecture, splits, effective sequence batch, epoch count, and seed while only
  the intended runtime controls differ.
- [ ] Record config hash, git SHA, seed, source checkpoint, split paths, non-PAD target
  count, optimizer-update count, wall time, peak MPS memory, skipped updates, and
  termination reason in each run manifest.
- [ ] Add a comparison command that rejects mismatched token exposure or provenance
  and emits both machine-readable JSON and a Markdown summary.
- [ ] Run smoke training and resume tests for both configs on MPS before full runs.

## Phase 2: Equal-Token Quality Validation
- [ ] Run one complete training-data traversal for reference and candidate at seed
  `1337`, followed by full validation, test, and generation sanity evaluation.
- [ ] Repeat the paired experiment at seed `2027` with identical evaluation settings.
- [ ] Produce `docs/mps_training_quality_validation.md` with per-seed and aggregate
  throughput, wall time, memory, loss, perplexity, termination, and genomic-quality
  results, plus exact reproduction commands.
- [ ] Apply the acceptance gates and record a pass/fail decision with supporting
  manifest paths. If necessary, execute and assess the conservative fallback.
- [ ] On pass, add a new recommended MPS config rather than rewriting historical
  experiment configs, and update benchmark/training documentation.

## Phase 3: Lossless Context Data and Token-Budget Training
- [ ] Replace suffix-only dynamic truncation for this experiment with deterministic
  one-token-overlap chunks of lengths 128, 256, and 512. The first chunk retains BOS,
  the last retains EOS, and every adjacent next-token transition appears exactly once.
- [ ] Add dataset tests that prove no split membership changes, no transition is lost
  or duplicated, and short sequences remain byte-for-byte equivalent after loading.
- [ ] Add a `target_tokens_per_update` training option that accumulates token-summed
  loss and normalizes gradients by the actual non-PAD target count before stepping.
- [ ] Persist accumulated-token and consumed-token counters in checkpoints and test
  exact mid-accumulation resume behavior.
- [ ] Support transfer into shorter absolute-position tables by loading the matching
  positional-embedding prefix while requiring all non-position backbone weights to
  match exactly.
- [ ] Build and audit 128-, 256-, and 512-token datasets from the same Stage 2.6
  source records and split assignment.

## Phase 4: Context-Length Ablation and Promotion
- [ ] Benchmark the largest stable physical batch for each context on the target
  M2/8 GB machine while holding `target_tokens_per_update` at 32,768 non-PAD targets.
- [ ] Train contexts 128, 256, and 512 for the same total non-PAD target budget using
  seeds `1337` and `2027`; vary only context length and the physical batch required
  for memory stability.
- [ ] Evaluate full validation/test loss and perplexity, natural termination, genomic
  quality/alignment, and prefix-to-function agreement using the same examples and
  decoding seeds for all contexts.
- [ ] Produce `docs/context_length_ablation.md` with quality, useful tokens/second,
  end-to-end wall time, peak memory, and exact reproduction commands.
- [ ] Select the shortest context whose mean validation loss is within 1%, downstream
  and genomic-quality metrics are within 5%, and termination is within 5 percentage
  points of context 512. Otherwise retain context 512.
- [ ] Promote the passing runtime and context choices to a new recommended training
  config, update the roadmap and track registry, and keep raw run artifacts ignored.

## Non-Goals
- Changing model depth, width, attention architecture, objectives, or vocabulary.
- Reinterpreting old experiment results by modifying their historical configs.
- Using validation subsets or smoke metrics as final scientific evidence.
- Running product-facing generation tracks before runtime quality parity is known.
