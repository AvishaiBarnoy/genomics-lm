# Model-Agnostic Training Engine Plan

## Phase 0: Characterize Existing Behavior

- [x] Inventory every training and fine-tuning entry point and classify its task,
  update algorithm, optimizer topology, scheduler, resume granularity, and outputs.
- [x] Add characterization tests for standard backpropagation, gradient
  accumulation remainders, nonfinite groups, validation selection, scheduler
  timing, interruption, and resume.
- [x] Define the versioned `TrainingTask`, `UpdateStrategy`, metric, callback, and
  engine-state contracts without importing CodonLM or ProteinLM modules.
- [x] Define compatibility rules for existing checkpoints and run directories.

Exit gate: the contracts can represent every current trainer without model-name
branches, and tests capture the behavior that migrations must preserve.

## Phase 1: Minimal Shared Engine

- [x] Implement a model-agnostic `TrainingEngine` over `TrainingRun`.
- [x] Implement standard accumulated-backprop and optimizer/scheduler strategies.
- [x] Add shared device, precision, clipping, nonfinite, wall-time, validation,
  metric aggregation, callback, and checkpoint policies.
- [x] Add engine contract tests with synthetic tasks and deterministic failures.

Exit gate: synthetic tasks prove exact fresh, partial-group, interrupted, resumed,
and completed behavior without importing a biological model.

## Phase 2: ProteinLM Reference Migration

- [x] Implement `ProteinLMTask` as the simplest causal-language-model adapter.
- [x] Demonstrate fixed-seed parity for updates, scheduler state, validation loss,
  checkpoint payload, and interrupted/resumed parameters.
- [x] Replace ProteinLM loop orchestration while retaining its CLI and legacy
  checkpoint compatibility.

Exit gate: ProteinLM behavior is equivalent and its trainer contains only argument,
configuration, task, strategy, and engine assembly.

## Phase 3: Supervised And Contrastive Protein Tasks

- [x] Add an end-of-phase metric hook so non-decomposable supervised metrics such
  as F1 are computed over the complete prediction set rather than averaged per batch.
- [x] Implement the bidirectional multitask `ProteinCriticTask`.
- [x] Implement the corrected single-label `ProteinClassifierTask`; retain its
  train-fitted label vocabulary as task-owned state rather than over-generalizing
  the multitask critic adapter.
- [x] Implement `ProteinEBMTask`, keeping positive/negative construction and energy
  metrics task-owned while using the shared optimization strategy.
- [x] Verify the frozen-backbone invariant, head-only updates, validation selection,
  versioned critic loading, legacy EBM aliases, and interrupted resume parity.
- [x] Verify ProteinCritic class/regression metric aggregation, validation
  selection, legacy checkpoint compatibility, and optimizer-boundary resume parity.
- [x] Verify ProteinClassifier complete-set accuracy/weighted-F1, label-map
  compatibility, scheduler state, legacy aliases, and interrupted resume parity.
Exit gate: protein trainers share orchestration without changing their architectures,
decoy distributions, losses, or scientific metrics.

## Phase 4: CodonLM Migration

- [ ] Implement `CodonLMTask` with packed-data masks, multi-offset heads,
  termination loss, biophysical extensions, and existing telemetry as task hooks.
- [ ] Characterize and preserve committed-token accounting, invalid-group recovery,
  MPS memory telemetry, periodic checkpoints, and exact mid-epoch resume.
- [ ] Run CPU parity tests and a bounded MPS fresh/resume smoke test.
- [ ] Remove the old CodonLM orchestration loop only after parity gates pass.

Exit gate: all declared CodonLM configurations use the shared engine without
changing their objective, update exposure, or checkpoint compatibility.

## Phase 5: Nonstandard Update Algorithms

- [x] Implement a layer-local `NoPropUpdateStrategy` with complete per-layer
  optimizer state and resume parity.
- [x] Evaluate whether frozen-backbone and discriminative-learning-rate behavior
  require strategies, optimizer factories, or parameter-group configuration.
- [x] Document how future training algorithms register without editing the engine.

Exit gate: NoProp demonstrates that the engine supports a genuinely different
update algorithm without model-specific branches.

## Phase 6: Enforcement And Cleanup

- [ ] Register every trainer and add CI contract coverage for fresh, collision,
  resume, interruption, and completion behavior.
- [ ] Reject new standalone training loops unless an explicit exemption documents
  why the engine contract is insufficient.
- [ ] Remove duplicated orchestration helpers and obsolete checkpoint writers.
- [ ] Update workflow, architecture, development-log, and extension documentation.
- [ ] Benchmark engine overhead and confirm it is negligible relative to model work.

Exit gate: reusable mechanics have one maintained implementation, all supported
trainers use it, and task-specific code contains only scientifically necessary
behavior.

## Future Scientific Features

- [ ] Add validated per-task ProteinCritic loss weights in a separate PR,
  preserving the present objective as the default and selecting weights using
  training/validation diagnostics rather than the test split.

## Planned Pull Requests

1. Contracts, trainer inventory, and characterization tests.
2. Minimal engine plus synthetic contract tests.
3. ProteinLM reference migration.
4. ProteinCritic, classifier, and EBM migrations.
5. CodonLM migration and MPS parity gate.
6. NoProp strategy.
7. CI enforcement, ancillary-trainer decisions, documentation, and cleanup.
