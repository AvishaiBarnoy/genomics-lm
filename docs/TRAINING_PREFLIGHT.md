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
