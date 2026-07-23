# Corrected Primary Training Configs

The corrected primary CodonLM is frozen in four versioned configs:

| Config | Purpose | Limit |
| --- | --- | --- |
| `corrected_primary_pilot_genome_seed1337_v2.yaml` | MPS pilot and resume gate | one epoch, 30 minutes per invocation |
| `corrected_primary_genome_seed1337_v2.yaml` | primary genome replicate 1 | 10 complete epochs |
| `corrected_primary_genome_seed2027_v2.yaml` | primary genome replicate 2 | 10 complete epochs |
| `corrected_primary_genus_seed1337_v2.yaml` | separate harder genus holdout | 10 complete epochs |

The two genome replicates differ only in random seed and run identity. They consume
the same frozen training transitions for 10 full epochs, with early stopping disabled,
so their committed non-PAD exposure is directly comparable. The genus run uses a
different grouped split and is reported separately rather than pooled with those
replicates.

All four configs define the basic 10-layer, 8-head, width-384 causal model. They
explicitly disable checkpoint transfer, GQA, RoPE, SwiGLU, shape guidance,
multi-offset heads, termination/replay objectives, and backbone freezing. The MPS
runtime is batch 4, accumulation 32, AMP, activation checkpointing, SDPA with the
separator mask, batch-aware NPY mmap, and no DataLoader workers.

## Contract Validation

Validate a config without loading the large local dataset:

```bash
python -m scripts.validate_primary_training_config \
  configs/corrected_primary_pilot_genome_seed1337_v2.yaml
```

Marked configs are also validated automatically before the trainer loads data or
constructs the model. Unknown keys and changes to data identity, architecture,
objective, optimizer, scheduler, runtime, exposure, or output identity are fatal.
CLI data, transfer-checkpoint, and run-ID overrides are rejected. An OOM saves
`last.pt` but never rewrites an immutable config; changing batch policy requires a
new reviewed contract version.

## Pilot

Start the bounded pilot on Apple Silicon:

```bash
caffeinate -i python -m src.codonlm.train_codon_lm \
  --config configs/corrected_primary_pilot_genome_seed1337_v2.yaml
```

The 30-minute limit is per process invocation. The pilot retains the full primary
5,000-step cosine horizon; its one-epoch bound does not compress the learning-rate
schedule. It intentionally exercises a
mid-epoch checkpoint. Resume from the committed accumulation-group boundary:

```bash
caffeinate -i python -m src.codonlm.train_codon_lm \
  --config configs/corrected_primary_pilot_genome_seed1337_v2.yaml \
  --resume runs/corrected-codonlm-v1-pilot-genome-seed1337/checkpoints/last.pt
```

Repeat the resume command until the one-epoch pilot completes validation and writes
`best.pt`. Approve the full configs only after checking finite loss, zero aborted
groups, optimizer/scheduler advancement, committed-token continuity, peak memory,
throughput, dataset/vocabulary provenance, and the observed epoch wall time.

Training-loss sums, counts, and the first finite loss are checkpointed with each
partial epoch, so the final epoch training loss covers all resumed segments rather
than only the last process invocation.

## Full Runs

After pilot approval, run each primary config without overrides:

```bash
caffeinate -i python -m src.codonlm.train_codon_lm \
  --config configs/corrected_primary_genome_seed1337_v2.yaml
```

Use the same `--resume runs/<run-id>/checkpoints/last.pt` pattern after an interrupted
full run. Each epoch performs full validation and writes `last.pt`; an improved epoch
also writes `best.pt`. A periodic `last.pt` is written approximately every 30 minutes.
