# Shared Training Run Lifecycle

## Objective

Prevent every model trainer from overwriting an existing or more advanced run by
centralizing run-directory allocation, locking, resume validation, logging paths,
checkpoint lineage, and completion state in `src/training/runtime.py`.

## Contract

- A fresh launch atomically creates the requested directory or the next `-rNNN`
  serial directory; it never writes into an existing artifact-bearing directory.
- A resume appends only to the checkpoint's existing run directory unless an
  explicit fork is requested with a new run ID.
- Appending requires the selected checkpoint to be the run's newest `last`
  checkpoint. A validation-selected `best` checkpoint is for evaluation or an
  explicit fork, not in-place resume.
- Configured epochs are the total target. The target must exceed completed
  checkpoint progress.
- A filesystem lock prevents concurrent writers.
- Completion is recorded atomically and makes the run immutable.
- Model-specific payloads, checkpoint names, objectives, and optimizer restoration
  remain owned by each trainer.

## Compatibility

Legacy checkpoints remain readable. When legacy progress cannot be interpreted
unambiguously, in-place resume fails with an actionable error instead of guessing.

