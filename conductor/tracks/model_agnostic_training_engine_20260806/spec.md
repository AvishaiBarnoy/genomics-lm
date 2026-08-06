# Model-Agnostic Training Engine

## Objective

Centralize reusable training mechanics so optimizer, accumulation, precision,
validation, logging, checkpoint, interruption, and resume improvements apply to
every compatible trainer once. Preserve model, data, objective, and scientific
semantics behind explicit task and update-strategy interfaces.

## Architecture

The shared engine owns orchestration but does not inspect model types or biological
tokens. A `TrainingTask` adapts model construction, data, loss, metrics, and
task-specific checkpoint state. An `UpdateStrategy` owns how parameters are
updated, allowing standard backpropagation, frozen-backbone training, contrastive
EBM training, and layer-local NoProp to coexist without model-name conditionals in
the engine.

The existing `TrainingRun` lifecycle remains responsible for run allocation,
locking, artifact paths, resume validation, and completion. The new engine consumes
that service rather than duplicating it.

## Required Boundaries

### Shared engine responsibilities

- epoch and microbatch iteration;
- optimizer-step and scheduler-step timing;
- gradient accumulation, remainder scaling, clipping, and nonfinite recovery;
- device and precision policy, including MPS behavior;
- validation cadence and metric aggregation;
- structured progress, throughput, memory, and curve logging;
- periodic, epoch, best, interruption, and completion checkpoints;
- wall-time interruption and exact resumable progress;
- callback execution and deterministic random-state restoration.

### Task responsibilities

- model and dataloader construction;
- batch movement and domain-specific masks;
- forward pass, loss components, and task metrics;
- causal, bidirectional, contrastive, or auxiliary-objective semantics;
- task-specific checkpoint payloads and compatibility adapters;
- scientific evaluation outside the optimization loop.

### Update-strategy responsibilities

- optimizer ownership and zeroing;
- microbatch backward behavior;
- accumulation-group commit behavior;
- scheduler ownership;
- strategy-specific state serialization and restoration.

## Non-Goals

- Do not merge CodonLM and ProteinLM architectures or tokenizers.
- Do not change datasets, splits, objectives, hyperparameters, or reported results.
- Do not require all algorithms to imitate a standard single-optimizer loop.
- Do not put `if model_type == ...` branches in the generic engine.
- Do not remove legacy checkpoint readers until migration parity is demonstrated.

## Compatibility And Acceptance

Every migration must use fixed-seed characterization tests to compare the old and
new paths for optimizer-step count, committed examples/tokens, accumulation
remainder scaling, scheduler position, losses and metrics, checkpoint contents,
and interrupted/resumed final parameters. Expected floating-point tolerances and
any intentional behavior changes must be declared before implementation.

Migration occurs one trainer at a time. The old implementation is removed only
after its task adapter passes fresh-run, interruption, resume, completion, and
artifact-contract tests on CPU. CodonLM and ProteinCritic also require a bounded MPS
smoke test before their old orchestration paths are retired.

