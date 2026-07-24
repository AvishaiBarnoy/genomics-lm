# Corrected Primary MPS Pilot

## Decision

Phase 1 of the corrected training program passed on 2026-07-23. The immutable
schema-v3 genome-holdout pilot is approved as the runtime and lifecycle template
for full primary training.

This decision approves the training contract and MPS execution path. It is not a
scientific model-quality result: the pilot covers one of the ten planned epochs,
and its validation set must not be used as a test-set result.

## Frozen Contract

- Git revision: `b934e0acc4a6b76bf0aff25907f26a22a34f81b8`
- Config: `configs/corrected_primary_pilot_genome_seed1337_v3.yaml`
- Config SHA-256:
  `8144f4da545b34d27b85849895bf1e392fd2be477a2841cad738c4d361c2e4c8`
- Dataset freeze:
  `1582505ae40445422711fa15918ee9c229caf84b1b3feba1a71f078259892249`
- Genome dataset:
  `da3dfce28b7a46b8640d75c7cb417c867137a99e004ea359d85784ff0c269db9`
- Model: random-initialized 10-layer, 8-head, width-384 basic CodonLM
- Objective: next-token prediction only
- Runtime: Apple M2 MPS, batch 4, accumulation 32, FP16 autocast,
  activation checkpointing, MHA/SDPA, separator mask, and NPY mmap
- Scheduler: cosine over the full primary horizon of 5,000 optimizer steps

## Execution

The 30-minute pilot limit intentionally exercised save/resume behavior. The epoch
completed across six invocations. After every intermediate stop, the committed
microbatch boundary exactly matched the cumulative metric boundary:

| Stop | Committed microbatches | Optimizer step | Committed tokens |
| ---: | ---: | ---: | ---: |
| 1 | 3,520 | 110 | 5,557,284 |
| 2 | 7,072 | 221 | 11,158,433 |
| 3 | 10,528 | 329 | 16,627,698 |
| 4 | 13,216 | 413 | 20,864,291 |
| 5 | 15,904 | 497 | 25,098,947 |
| Final | 15,996 | 500 | 25,238,438 |

Each resume deterministically skipped already committed loader batches. Seen but
uncommitted microbatches at a wall-time stop were excluded from both token and loss
metrics and were recomputed after resume.

## Results

| Metric | Result |
| --- | ---: |
| Initial next-token loss | 223.3445 |
| Cumulative epoch train loss | 18.1526 |
| Full validation loss | 3.9338 |
| Validation perplexity | 51.101 |
| Final learning rate | 0.000295585 |
| Non-finite microbatches | 0 |
| Aborted accumulation groups | 0 |
| Peak MPS allocated memory | 1.16 GB |
| Peak MPS driver memory | 2.45 GB |
| Total segmented wall time | 9,349.5 seconds |

Both `best.pt` and `last.pt` were created locally. Run checkpoints remain ignored
because each file is about 226 MB; their hashes and the complete compact evidence
are recorded in
`docs/benchmarks/corrected_primary_pilot_genome_seed1337.json`.

## Acceptance

The pilot passed:

- exact committed token, optimizer, scheduler, and metric counters;
- full validation and best/last checkpoint creation;
- exact resume across five wall-time boundaries;
- finite training with no aborted accumulation group;
- stable host and MPS memory;
- scheduler alignment with the ten-epoch primary contract.

The next authorized step is Phase 2: start the immutable full
genome-held-out seed-1337 run from random initialization.
