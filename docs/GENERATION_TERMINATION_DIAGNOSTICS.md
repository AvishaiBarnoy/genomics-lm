# Generation Termination Diagnostics

## Purpose

The corrected base model beat the trigram PPL baseline but produced no natural
stops under temperature `0.8`, top-k `5` sampling. These diagnostics separate a
missing termination signal from a decoding policy that removes low-ranked stop
tokens.

## Stop Probability

Immediately before the true terminal codon in 50 held-out CDSs:

| Checkpoint seed | Mean stop + EOS probability | Median termination rank | Top-20 inclusion |
| --- | ---: | ---: | ---: |
| 1337 | 0.447% | 61 | 0% |
| 2027 | 0.317% | 61.5 | 0% |

The probability does not rise sharply from 32 codons before the terminal position
to the position immediately before it. EOS probability alone is approximately
`0.007-0.011%`.

On generated contexts, median termination rank improves with length but remains
outside the top 20. Therefore top-k 5 and top-k 20 assign exactly zero sampling
probability to all termination tokens in these samples.

## Decoding Pilot

Ten balanced held-out prompts were sampled per variant and checkpoint:

| Seed | Decoder | Natural stop | Hard cap | Mean generated codons | Mean GC |
| --- | --- | ---: | ---: | ---: | ---: |
| 1337 | T=0.8, unrestricted | 1/10 | 9/10 | 299.3 | 55.3% |
| 1337 | T=1.0, unrestricted | 9/10 | 1/10 | 176.9 | 54.1% |
| 1337 | T=1.0, top-k 20 | 0/10 | 10/10 | 300.0 | 70.9% |
| 2027 | T=0.8, unrestricted | 0/10 | 9/10 | 275.3 | 64.2% |
| 2027 | T=1.0, unrestricted | 9/10 | 1/10 | 191.6 | 61.2% |
| 2027 | T=1.0, top-k 20 | 0/10 | 10/10 | 300.0 | 60.9% |

The held-out prompt sources have mean GC `52.9%`. The pilot is too small to select
a final decoder, but it demonstrates that temperature `1.0` without top-k
truncation substantially restores natural stopping and reduces the extreme GC
drift seen under top-k 5.

## Exhaustive Novelty Audit

The memory-bounded audit now completes against all 74,600 training CDSs:

- Minimap2 performs the nucleotide search against the complete training FASTA.
- MMseqs2 searches translated proteins in 5,000-record target batches and retains
  the best hit over every batch.
- Exact substring coverage scans the training corpus once against only generated
  query windows, avoiding a materialized training-window index.

Neither alignment tool reported a qualifying nearest hit for these highly
divergent generated sequences under its default sensitivity. This means no
alignment was reported, not that identity is zero. Exact 30-nt coverage is at most
`3.32%`; mean exact 10-aa coverage ranges from `5.0%` to `10.2%` across checkpoint
and protocol, with a maximum of `28.8%`.

## Decision

The base model contains a weak termination signal, but truncating the distribution
at top-k 5 or 20 removes it. The next controlled generation run should use
unrestricted temperature `1.0` as the decoder baseline. Termination-head and replay
training remain justified only if a larger unrestricted evaluation still shows
unacceptable length calibration, early stops, or hard caps.

Decoder correction does not prove protein quality. ORF, composition, diversity,
and novelty controls remain required for the larger run.
