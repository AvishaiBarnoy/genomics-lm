# Corrected MPS Runtime Gate

## Decision

The corrected primary training policy retains batch 4, gradient accumulation 32,
activation checkpointing, FP16 MPS autocast, standard multi-head SDPA, and the
explicit separator mask. It adopts the merged batch-aware NPY mmap loader.

No optimized compute candidate entered the equal-token quality phase. The
predeclared runtime gate required at least a 1.5x speedup before quality promotion:
checkpointing off at batch 4 reached only 1.31x and increased MPS driver memory by
about 47%; batch 8 was slower than the reference and approached the unified-memory
pressure regime. Retaining the reference is the fail-closed outcome, not evidence
that checkpointing improves model quality.

## Target And Protocol

- Hardware: Apple M2 with 8 GB unified memory.
- Software: macOS 26.5.1, PyTorch 2.12.0, MPS backend.
- Dataset: frozen corrected genome holdout
  `da3dfce28b7a46b8640d75c7cb417c867137a99e004ea359d85784ff0c269db9`.
- Model: random-initialized basic 10-layer, 8-head, width-384 CodonLM.
- Objective: next-token prediction only.
- Exposure: 20 warmup and 100 measured microbatches, seed 1337.
- Every variant used effective sequence batch 128.

```bash
caffeinate -i python -m scripts.benchmark_training_speed \
  --matrix configs/corrected_mps_runtime_matrix.yaml \
  --out runs/corrected_mps_runtime_gate
```

## Results

| Variant | Useful tok/s | Dataset RSS delta | Peak MPS driver | Estimated epoch |
| --- | ---: | ---: | ---: | ---: |
| Preloaded, b4, checkpoint | 2,892 | 508.6 MB | 2.45 GB | 142.1 min |
| Mmap, b4, checkpoint | 2,900 | 1.1 MB | 2.45 GB | 141.7 min |
| Mmap, b4, no checkpoint | 3,807 | 0.6 MB | 3.59 GB | 107.9 min |
| Mmap, b8, no checkpoint | 1,105 | 0.8 MB | 4.71 GB | 375.9 min |

Batch-aware mmap was throughput-neutral (`1.003x`) and reduced dataset-loading RSS
by `99.79%`. This is the only promoted change. The raw machine-readable evidence is
stored in `docs/benchmarks/corrected_mps_runtime_gate.json`.

The epoch estimate is a throughput projection, not a completed training epoch. Full
quality measurement begins with the bounded corrected-data pilot after immutable
primary configs pass their contract tests.
