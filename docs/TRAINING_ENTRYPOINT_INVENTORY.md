# Training Entrypoint Inventory

This inventory defines the migration surface for the model-agnostic training
engine. `docs/training_entrypoints.toml` is the machine-readable CI source of truth:
every runnable `train_*.py` entrypoint must be registered. A production trainer
produces durable model parameters for scientific use. Diagnostic harnesses may
execute optimizer steps but do not own durable run semantics.

| Entrypoint | Task | Update topology | Resume | Migration |
| --- | --- | --- | --- | --- |
| `src/codonlm/train_codon_lm.py` / `src/codonlm/training/loop.py` | causal codon LM with optional auxiliary objectives | AdamW, accumulated backprop, scheduler per committed group | exact optimizer boundary | Phase 4 |
| `src/protein_lm/train_lm.py` | causal amino-acid LM | AdamW, accumulated backprop, cosine scheduler | optimizer boundary | Phase 2 reference |
| `src/protein_lm/train_classifier.py` | protein sequence classifier | shared engine: configurable optimizer, accumulated backprop, cosine scheduler | optimizer boundary | Phase 3 migrated |
| `src/protein_lm/train_multi_task.py` | bidirectional multitask ProteinCritic | AdamW, accumulated backprop, mixed classification/regression loss | optimizer boundary | Phase 3 |
| `src/protein_lm/train_ebm.py` | latent real-versus-corrupted ranking | shared engine: AdamW on EBM head; frozen critic | optimizer boundary | Phase 3 migrated |
| `src/codonlm/train_noprop.py` | layer-local NoProp codon model | shared engine: embedding, per-block, and head optimizers committed by `NoPropUpdateStrategy` | optimizer boundary | Phase 5 migrated |
| `src/protein_lm/train_mlp_heads.py` | Pfam/EC/stability classifiers over frozen feature arrays | shared engine: one AdamW optimizer with independent head parameters | optimizer boundary | ancillary migrated |
| `scripts/train_biophysics_fusion.py` | synthetic DNA-shape encoder pretraining plus optional fusion smoke test | shared engine: AdamW encoder regression | optimizer boundary | ancillary migrated |

`scripts/train_classifier.py` is a registered exemption: it evaluates downstream
embedding and k-mer baselines. Its optional MLP probe uses a small fit helper and
does not expose production run lifecycle or resume semantics. The CodonLM trainer
is the only deferred production trainer; the registry requires an explicit reason
for both statuses. `tests/test_training_entrypoint_registry.py` makes new or
unclassified `train_*.py` files fail CI and preserves the shared fresh-run,
collision, resume, interruption, and completion contract coverage.

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

## Registering Update Algorithms

A trainer whose only differences are frozen parameters, optimizer type, or
discriminative learning rates should use the standard accumulated-backprop strategy
with appropriate optimizer parameter groups. A custom `UpdateStrategy` is reserved
for algorithms that change update timing or gradient flow. NoProp is the reference:
its strategy owns the embedding, per-block, and head optimizers, while its task owns
the layer-local denoising objectives and explicit detach boundaries. The generic
engine requires no model-name branch.

The feature-head trainer treats `--out_dir` as a collision-safe run root. Each run
stores versioned `last.pt`/`best.pt` checkpoints, the selected legacy-format
`checkpoints/mlp_heads.pt`, per-head loss curves, and a run log. Use `--resume` with
the newest `last.pt` and the allocated `--run_id`; periodic and wall-time controls
are available through `--checkpoint_every_steps` and `--max_time_minutes`.

The biophysics script trains only the nucleotide encoder; it does not fine-tune the
CodonLM generator. It stores a validation-selected raw encoder artifact alongside
versioned engine checkpoints. Pass `--generator-run` only to perform the optional
post-training fusion smoke test. Shape-guided CodonLM training can select either the
raw artifact or a versioned engine checkpoint through
`biophysics_encoder_checkpoint`; the historical `runs/biophysics_encoder.pt` path
remains a fallback for legacy configurations.

The encoder protocol defaults live in `configs/biophysics_encoder.yaml`. The total
synthetic corpus is generated once from its seed, then split by the configured
train/validation/test fractions. The resolved counts and settings are copied into
each run directory, and the held-out test split is reported only after training.

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
