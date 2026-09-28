# Corrected Training Preflight

Run this gate before freezing datasets or starting a long training job. It creates
a tiny manifest-validated dataset, trains through the real CLI, saves a checkpoint,
restarts the process, resumes for another epoch, and validates optimizer, scheduler,
committed-token, vocabulary, dataset-identity, accumulation-health, and memory state.

CPU integration, also executed in CI:

```bash
python -m scripts.training_preflight \
  --device cpu \
  --work-dir /tmp/codonlm-preflight-cpu
```

Apple Silicon MPS integration:

```bash
python -m scripts.training_preflight \
  --device mps \
  --work-dir /tmp/codonlm-preflight-mps
```

Explicit device requests never fall back. `--device mps` fails when MPS is not
available. The report is written to `<work-dir>/preflight_report.json`; child
training logs and checkpoints remain under the same isolated directory.

## Passing Gate

A pass requires the requested and actual devices to match, optimizer and scheduler
steps to advance after restart, committed non-PAD tokens to increase, dataset and
vocabulary identities to remain unchanged, and all non-finite/aborted accumulation
counters to remain zero. Resume on a different manifest identity is fatal.

On 2026-07-21, the host M2 MPS run passed with 2 optimizer steps and 40 committed
tokens before restart, 4 steps and 80 tokens after resume, zero invalid groups,
approximately 90 KB peak live MPS tensor allocation, approximately 20.9 MB peak MPS
driver allocation, and 4.43 seconds total preflight wall time. These figures validate
the lifecycle only; they are not training-throughput or model-quality measurements.

The lifecycle gate was repeated after shared-engine enforcement on 2026-09-28. It
passed on MPS with optimizer steps advancing from 2 to 4, committed tokens from 40
to 80, and scheduler steps from 2 to 4. All non-finite and aborted-group counters
remained zero. Peak live MPS tensor allocation was 96,768 bytes, peak MPS driver
allocation was 21,266,432 bytes, and total preflight wall time was 8.73 seconds on
PyTorch 2.12.0. This validates the legacy CodonLM lifecycle path. It does not by
itself validate the shared `TrainingEngine` fork path or complete that track's
acceptance gate.

The 2026-09-28 corrective preflight extended the same command to exercise a
shared-engine full-state fork on Apple MPS. The source run completed one epoch and
two optimizer steps; the fork restored that state, completed a second epoch, and
reached four optimizer steps. Its `run_lineage.json` identified distinct `source`
and `fork` run IDs and matching immutable configuration fingerprints. The combined
preflight passed in 4.89 seconds on PyTorch 2.12.0. The lifecycle track remains open
because the primary CodonLM trainer is still listed as deferred rather than migrated
to `TrainingEngine`.
