# Suite Runner Main.sh Evolution Plan

This track turns `main.sh` from a CodonLM-oriented wrapper into an explicit suite runner for CodonLM and ProteinLM training workflows.

The current `main.sh` is now suitable as the preferred CodonLM entrypoint, including resume, dataset-prep bypass, run logging, and batch-optimizer dispatch. It is not yet suitable for ProteinLM because it assumes CodonLM data-prep, CodonLM evaluation, and `src.codonlm.train_codon_lm`.

---

## Status

- **State:** Completed
- **Completed:** 2026-07-06
- **Primary file:** `main.sh`
- **Related tracks:**
  - `codonlm_trainer_refactor_20260622`
  - `training_speed_optimization_20260615`
  - `long_range_codon_objectives_20260616`

---

## Problem

Long local runs now need a single reliable orchestration entrypoint, but `main.sh` is still implicitly CodonLM-specific:

- It assumes CodonLM packed NPZ data and CodonLM data preparation.
- It dispatches only to `scripts.optimize_train_batching` or `src.codonlm.train_codon_lm`.
- It runs CodonLM perplexity and mutation-scoring post-train steps.
- It has no explicit trainer type for ProteinLM, ProteinCritic, or future suite jobs.

This makes the current wrapper useful for CodonLM but unsafe to reuse for ProteinLM.

---

## Design Goal

Make `main.sh` a conservative suite runner that:

- Reads a trainer/task type from config.
- Dispatches only to known supported trainers.
- Keeps CodonLM behavior backward-compatible.
- Adds ProteinLM support explicitly, not by accidental inference.
- Preserves resume, run-id, logging, checkpoint, and no-sleep workflows.
- Avoids interactive prompts so long runs can be unattended.

---

## Proposed Config Contract

Add one of these equivalent config keys:

```yaml
trainer: codon_lm
```

or:

```yaml
task:
  trainer: codon_lm
```

Supported initial values:

- `codon_lm`
- `protein_lm`
- `protein_multitask`
- `protein_classifier`

Backward compatibility:

- If no trainer key exists, default to `codon_lm` for existing configs.
- Print the resolved trainer at startup.
- Fail fast on unknown trainer values.

---

## Phase 0: Baseline Behavior Lock

- [ ] Document current CodonLM `main.sh` behavior:
  - [ ] direct training when `batch_optimizer` absent or disabled
  - [ ] optimizer dispatch when `batch_optimizer.enabled`
  - [ ] resume run-id derivation from `runs/<RUN_ID>/checkpoints/last.pt`
  - [ ] prepared NPZ pass-through
- [ ] Add shell syntax validation to CI or local checks:
  - [ ] `bash -n main.sh`
- [ ] Add a small dry-run mode or dispatch-print test for trainer resolution.

Acceptance criteria:

- Existing CodonLM command still works:

```bash
caffeinate -i ./main.sh \
  --config configs/physical_termination_transfer.yaml \
  --resume runs/2026-06-19_physical_termination_transfer_mps_b4_e1/checkpoints/last.pt
```

---

## Phase 1: Explicit Trainer Resolution

- [ ] Add a helper script or inline Python block to resolve trainer type from config.
- [ ] Default missing trainer to `codon_lm`.
- [ ] Print:
  - config path
  - resolved trainer
  - run id
  - resume path
  - whether batch optimizer is active
- [ ] Fail on unknown trainer before any data prep starts.

Acceptance criteria:

- CodonLM configs without `trainer` keep working.
- Unknown trainer configs fail early with a clear error.

---

## Phase 2: CodonLM Dispatch Cleanup

- [ ] Move CodonLM-specific data prep into a clearly named branch/function.
- [ ] Keep current behavior for:
  - config-provided `train_npz`
  - pipeline data preparation
  - `batch_optimizer` dispatch
  - direct `train_codon_lm` dispatch
  - post-train CodonLM perplexity/mutation scoring
- [ ] Ensure `--force` semantics are documented:
  - force data prep
  - force batch optimization if batch optimizer is enabled

Acceptance criteria:

- CodonLM path remains behavior-compatible with current `main.sh`.
- Batch optimizer cache is reused unless `--force` or config force is set.

---

## Phase 3: ProteinLM Dispatch

- [ ] Add `trainer: protein_lm` branch.
- [ ] Dispatch to `src.protein_lm.train_lm`.
- [ ] Pass `--resume` if supported by the script.
- [ ] Preserve run logging under `runs/<RUN_ID>/`.
- [ ] Skip CodonLM data prep/evaluation/mutation scoring for ProteinLM.
- [ ] Add a ProteinLM config example with explicit `trainer: protein_lm`.

Acceptance criteria:

- ProteinLM training can be launched through `main.sh`.
- ProteinLM path does not call CodonLM data prep or CodonLM eval.
- Existing direct ProteinLM script invocation remains supported.

---

## Phase 4: Protein Multi-Task and Classifier Dispatch

- [ ] Add `trainer: protein_multitask` branch for `src.protein_lm.train_multi_task`.
- [ ] Add `trainer: protein_classifier` branch for `src.protein_lm.train_classifier`.
- [ ] Preserve each script's existing config semantics.
- [ ] Keep batch optimization disabled for ProteinLM unless a future protein-specific optimizer exists.

Acceptance criteria:

- Multi-task critic and classifier training have suite-runner launch paths.
- No CodonLM-only assumptions leak into ProteinLM branches.

---

## Phase 5: Evaluation and Post-Processing Policy

- [ ] Split post-train actions by trainer type.
- [ ] CodonLM:
  - perplexity eval
  - mutation scoring when primary DNA exists
  - optional motifs/artifacts
- [ ] ProteinLM:
  - define minimal eval hooks or skip by default with a clear log line
- [ ] Protein critic/classifier:
  - define metrics collection path if available

Acceptance criteria:

- Post-train behavior is explicit and trainer-specific.
- Unsupported post-processing is skipped with an informative log, not attempted.

---

## Phase 6: Documentation and Migration

- [ ] Update `docs/MANUAL.md` with the suite-runner contract.
- [ ] Update `README.md` quickstart if needed.
- [ ] Add examples:
  - CodonLM with batch optimizer
  - CodonLM direct
  - ProteinLM
  - Protein multi-task critic
- [ ] Add notes about when to bypass `main.sh` and call a trainer directly.

Acceptance criteria:

- Users can tell when `main.sh` is preferred and when direct trainer invocation is appropriate.

---

## Validation Plan

Run after each phase:

```bash
bash -n main.sh
ruff check scripts/optimize_train_batching.py tests/test_batch_optimizer.py
pytest -q tests/test_batch_optimizer.py tests/test_wall_time.py tests/test_multi_task_wall_time.py
```

For shell dispatch changes, add or run lightweight dry-run tests once available:

```bash
./main.sh --config configs/physical_termination_transfer.yaml --dry-run
./main.sh --config configs/protein_lm_example.yaml --dry-run
```

---

## Open Questions

- Should `main.sh` stay Bash, or should it become a Python suite runner after trainer dispatch grows?
- Should batch optimization remain CodonLM-only, or should we create a generic optimizer interface later?
- Should `RUN_ID` always be explicit for resume, or is deriving it from checkpoint path sufficient?
- Should no-sleep behavior be inside `main.sh`, or should users continue wrapping with `caffeinate -i`?

---

## Done Definition

- `main.sh` resolves trainer type explicitly.
- CodonLM behavior remains backward-compatible.
- ProteinLM can be launched without CodonLM data prep/eval side effects.
- Protein multi-task and classifier training have explicit dispatch paths or documented deferrals.
- Documentation describes the supported suite-runner contract.
