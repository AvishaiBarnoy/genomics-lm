# CodonLM Trainer Refactor Plan

This track captures the staged refactor of `src/codonlm/train_codon_lm.py` after the batch optimizer, wall-time, and mid-epoch resume work exposed that the trainer has become too broad.

The goal is not to change training behavior. The goal is to split runtime orchestration, checkpoint/resume, data setup, objective computation, and CLI wiring into testable modules while preserving current checkpoints and configs.

---

## Status

- **State:** Completed
- **Completed:** 2026-07-06
- **Primary file:** `src/codonlm/train_codon_lm.py`

---

## Current Problem

`train_codon_lm.py` currently owns too many responsibilities:

- CLI parsing and config normalization.
- Dataset loading, audit, dynamic collation, and deterministic/bucketed dataloader setup.
- Model construction and transfer-learning vocab adaptation.
- Optimizer and scheduler creation.
- Objective computation for next-token, multi-offset, and termination-auxiliary losses.
- Runtime concerns: device selection, MPS behavior, wall-time limits, logs, periodic checkpoints, and failure metadata.
- Checkpoint serialization, resume loading, and mid-epoch resume semantics.
- Epoch loop, validation, best/last checkpoint policy, curves, and metrics.

This makes new features risky because small changes to one responsibility can break unrelated training behavior.

---

## Design Principles

- Preserve current public commands and config keys.
- Preserve compatibility with existing checkpoints whenever possible.
- Keep `train_codon_lm.py` as a thin CLI wrapper by the end of the track.
- Extract one responsibility at a time and run tests after each phase.
- Add tests around behavior before moving code that lacks coverage.
- Keep suite-level orchestration such as batch optimization and end-to-end wall-time budgeting outside the core trainer.

---

## Phase 0: Baseline Lock

- [ ] Record the current direct-training and optimizer-training commands used for physical termination transfer.
- [ ] Add or update smoke tests for:
  - [ ] Direct `train_codon_lm` wall-time checkpoint.
  - [ ] Resume from `last.pt`.
  - [ ] Mid-epoch resume metadata.
  - [ ] Transfer loading across vocab expansion.
  - [ ] Batch optimizer selected-config launch path.
- [ ] Confirm existing focused suite passes before extraction.

Acceptance criteria:

- No production code movement yet.
- Tests clearly describe the behavior that must survive the refactor.

---

## Phase 1: Checkpoint and Resume Module

- [ ] Create a dedicated checkpoint/resume module, likely `src/codonlm/checkpoints.py`.
- [ ] Move payload construction, atomic save calls, compatibility reads, and resume warnings into that module.
- [ ] Keep support for old checkpoints without `epoch_microbatch_idx`.
- [ ] Preserve safe behavior when resuming with changed `batch_size` or `grad_accum_steps`.

Acceptance criteria:

- `last.pt` and `best.pt` payloads are unchanged except for intentional metadata fields.
- Old checkpoints can still restore model weights.
- New checkpoints preserve mid-epoch resume metadata.

---

## Phase 2: Data Setup Module Boundary

- [ ] Keep dataset classes and dataloader construction in `src/codonlm/data_loading.py`.
- [ ] Move trainer-specific data setup into a small helper:
  - normalize train/val/test paths
  - build datasets
  - run length audit
  - build deterministic epoch loaders
- [ ] Ensure bucket sampler seed behavior remains reproducible per epoch.

Acceptance criteria:

- Dynamic and fixed NPZ tests still pass.
- Bucketed and non-bucketed loaders preserve current behavior.
- Resume skip semantics remain deterministic.

---

## Phase 3: Objective Computation Module

- [ ] Extract next-token, multi-offset, and termination-auxiliary objective computation into a focused helper.
- [ ] Keep loss weighting and label smoothing behavior identical.
- [ ] Keep MPS autocast fallback outside the pure objective helper.

Acceptance criteria:

- Existing long-range objective tests pass.
- Termination auxiliary labels and loss tests pass.
- Training curves remain comparable on smoke configs.

---

## Phase 4: Runtime Loop Extraction

- [ ] Introduce a trainer state object for:
  - epoch
  - optimizer step
  - best validation state
  - resume microbatch index
  - current microbatch index
- [ ] Move `one_pass` and epoch loop into a trainer helper/class.
- [ ] Keep logging and metrics emission compatible with existing `curves.csv` and `metrics.json`.

Acceptance criteria:

- Direct training command still works.
- Periodic checkpoints still write `last.pt`.
- Wall-time stop still writes `last.pt` and `meta.json`.
- The run log still captures stdout/stderr and crash records.

---

## Phase 5: CLI Thin Wrapper

- [ ] Reduce `train_codon_lm.py` to:
  - parse args
  - load config
  - normalize run id and paths
  - call the trainer entrypoint
- [ ] Keep module invocation stable:
  - `python -m src.codonlm.train_codon_lm --config ...`
- [ ] Document the internal module boundaries in `docs/MANUAL.md`.

Acceptance criteria:

- Existing user commands do not change.
- Tests do not need special-case imports from the old trainer file except legacy exports intentionally retained.

---

## Phase 6: Suite-Level Runtime Policy

- [ ] Decide which runtime policies remain in the trainer and which belong to `scripts.optimize_train_batching.py` or `main.sh`.
- [ ] Keep end-to-end wall-time budgeting in the suite runner.
- [ ] Keep trainer-local checkpoint-on-stop behavior for direct runs and backward compatibility.
- [ ] Document precedence between:
  - config `max_time_minutes`
  - optimizer `include_in_wall_time`
  - selected-config remaining time

Acceptance criteria:

- Direct trainer runs remain useful.
- Optimized runs treat wall-time as an end-to-end budget.
- No unattended command blocks on interactive prompts.

---

## Validation Plan

Run after each phase:

```bash
python -m py_compile src/codonlm/train_codon_lm.py src/codonlm/data_loading.py scripts/optimize_train_batching.py
ruff check src/codonlm/train_codon_lm.py src/codonlm/data_loading.py scripts/optimize_train_batching.py tests/test_batch_optimizer.py tests/test_dynamic_dataset.py tests/test_wall_time.py
pytest -q tests/test_batch_optimizer.py tests/test_dynamic_dataset.py tests/test_wall_time.py tests/test_long_range_codon_objectives.py tests/test_transfer_learning.py
```

Run before closing:

```bash
pytest -q
```

If the full suite is too slow on local MPS, run the focused suite locally and let CI run the full suite.

---

## Open Questions

- Should `max_time_minutes` eventually be removed from trainer configs, or kept as a direct-run convenience?
- Should mid-epoch resume skip metadata become strict, failing if dataset identity changes, or remain warning-only?
- Should `main.sh` become the preferred public entrypoint for all long training runs?
- Should CodonLM and ProteinLM share a common trainer runtime after this refactor, or stay separate with shared utilities only?

---

## Done Definition

- `train_codon_lm.py` is a thin wrapper rather than a monolithic trainer.
- Checkpoint/resume behavior is covered by tests.
- Mid-epoch resume is deterministic for future checkpoints.
- Optimized training and direct training keep their current user-facing commands.
- The physical termination transfer run can resume safely after periodic or wall-time checkpointing.
