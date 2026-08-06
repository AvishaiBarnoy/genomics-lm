# Shared Training Run Lifecycle Plan

## Phase 1: Runtime Contract

- [x] Add atomic serial-directory allocation and active-run locking.
- [x] Add standard checkpoint, score, log, metadata, and completion paths.
- [x] Add generic progress and newest-`last` resume validation.
- [ ] Add immutable-setting fingerprints and explicit fork semantics. Immutable
  fingerprints are implemented; explicit checkpoint forks remain open.
- [x] Test duplicate, concurrent, completed, stale-best, epoch-target, and valid
  resume cases.

## Phase 2: Primary Trainers

- [x] Migrate CodonLM without changing its existing checkpoint fields or scientific logic.
- [x] Migrate multitask ProteinCritic and preserve `last_critic.pt` compatibility.
- [x] Verify fresh serial allocation, mid-epoch resume, completion, and logging on CPU.
  A bounded MPS lifecycle preflight remains required before global enforcement.

## Phase 3: Remaining Model Trainers

- [x] Migrate protein LM and protein classifier with deterministic mid-epoch resume.
- [x] Add epoch-boundary resume support and migrate Protein EBM.
- [x] Migrate NoProp, including all layer-specific optimizer states, and explicitly
  document its experimental checkpoint contract.
- [ ] Inventory ancillary fine-tuning entry points and either migrate or mark them
  non-resumable with overwrite protection.

## Phase 4: Documentation and Enforcement

- [x] Document fresh, resume, fork, and completed-run commands.
- [ ] Add CI contract tests covering every registered trainer.
- [ ] Record migration status in the development log.
- [ ] Reject direct unguarded writes to canonical `last`/`best` paths in trainers.
