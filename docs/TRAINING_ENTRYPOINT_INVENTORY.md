# Training Entrypoint Inventory

This inventory defines the migration surface for the model-agnostic training
engine. A registered trainer produces durable model parameters for scientific use.
Diagnostic harnesses may execute optimizer steps but are not production trainers.

| Entrypoint | Task | Update topology | Resume | Migration |
| --- | --- | --- | --- | --- |
| `src/codonlm/train_codon_lm.py` / `src/codonlm/training/loop.py` | causal codon LM with optional auxiliary objectives | AdamW, accumulated backprop, scheduler per committed group | exact optimizer boundary | Phase 4 |
| `src/protein_lm/train_lm.py` | causal amino-acid LM | AdamW, accumulated backprop, cosine scheduler | optimizer boundary | Phase 2 reference |
| `src/protein_lm/train_classifier.py` | protein sequence classifier | shared engine: configurable optimizer, accumulated backprop, cosine scheduler | optimizer boundary | Phase 3 migrated |
| `src/protein_lm/train_multi_task.py` | bidirectional multitask ProteinCritic | AdamW, accumulated backprop, mixed classification/regression loss | optimizer boundary | Phase 3 |
| `src/protein_lm/train_ebm.py` | latent real-versus-corrupted ranking | AdamW on EBM head; frozen critic | epoch boundary | Phase 3 |
| `src/codonlm/train_noprop.py` | layer-local NoProp codon model | embedding, per-block, and head optimizers | epoch boundary | Phase 5 |
| `src/protein_lm/train_mlp_heads.py` | standalone heads over frozen features | one AdamW optimizer per head | none | ancillary; migrate or explicitly guard |
| `scripts/train_biophysics_fusion.py` | nucleotide biophysics encoder pretraining/fusion assembly | AdamW encoder pretraining | none | ancillary; migrate or explicitly guard |

## Diagnostic And Library Code

- `scripts/benchmark_protein_critic_training.py`, `scripts/optimize_train_batching.py`,
  and `scripts/profile_train.py` are diagnostic harnesses. They should call registered
  tasks or strategies where practical, but do not own durable run semantics.
- `scripts/analyze_saliency.py` performs an optimization-based interpretation
  procedure, not model training.
- `src/classifiers/mlp_head.py` is reusable library code invoked by evaluation
  workflows rather than a standalone run owner.

## Existing Characterization Coverage

- `tests/test_trainer_utils.py` fixes accumulation remainder scaling and nonfinite
  group semantics.
- `tests/test_nonfinite_accumulation.py` fixes abort, checkpoint, and resume counters.
- `tests/test_training_run_lifecycle.py` fixes collision, completion, curve, and
  newest-last resume behavior.
- `tests/test_training_preflight.py`, `tests/test_wall_time.py`, and
  `tests/test_multi_task_wall_time.py` fix interruption and resume behavior.
- `tests/test_protein_trainer_lifecycle.py` fixes ProteinLM serial allocation and
  optimizer-boundary resume behavior.

The migration parity suite will build on these tests and add fixed-seed parameter
comparisons for each trainer before its existing orchestration loop is removed.

## Checkpoint Compatibility Rules

New engine checkpoints use a versioned, namespaced envelope with `engine`, `task`,
`strategy`, `rng`, and `metadata` sections. The engine owns progress only; a task
owns model and objective state; an update strategy owns every optimizer and
scheduler it controls. This prevents the engine from interpreting model-specific
keys.

Existing checkpoints remain inputs to their current trainers during migration.
Each task adapter may provide an explicit legacy translator for a documented schema.
The generic engine must never infer ambiguous legacy progress or silently omit an
optimizer. A translated checkpoint is written in the new schema only after all
required state validates. Evaluation-only checkpoints remain usable for evaluation
and explicit transfer, but they do not become in-place resume checkpoints.
