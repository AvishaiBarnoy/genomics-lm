# Unrestricted Generation Confirmation

## Protocol

The exploratory decoder pilot suggested that temperature `1.0` without top-k
truncation might restore natural termination. The confirmatory evaluation used:

- both corrected base-model checkpoints;
- 50 held-out CDS prompts per checkpoint, balanced 25/25 across two genomes;
- the first natural codon as context;
- one deterministic sample per prompt;
- unrestricted vocabulary sampling at temperature `1.0`;
- a maximum of 300 generated tokens;
- exhaustive train-only nucleotide, protein, and exact-window novelty auditing.

No forced stop, CDS mask, critic, termination bias, or replay intervention was used.

## Results

| Seed | Natural stop | 95% Wilson CI | EOS | Hard cap | Mean codons | Median codons | Mean GC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1337 | 35/50 (70%) | 56.2-80.9% | 0/50 | 15/50 (30%) | 175.6 | 162.0 | 52.5% |
| 2027 | 28/50 (56%) | 42.3-68.8% | 1/50 | 21/50 (42%) | 205.5 | 248.5 | 58.4% |

The selected held-out source CDSs have mean length 350.1 codons, median length
294 codons, and mean GC 52.9%. Among naturally stopped generations, median lengths
are 119 codons for seed 1337 and 152 codons for seed 2027. Because the prompt
contains only the first codon, paired generated and source lengths are not expected
to match exactly; nevertheless, the combination of short completed sequences and
30-42% hard caps shows poor length calibration.

The earlier 10-prompt pilot reported 9/10 natural stops for each seed. The larger
result demonstrates why that pilot was insufficient for promotion.

## Novelty

The exhaustive bounded audit searched all 74,600 pretraining CDSs:

- no qualifying nucleotide or protein alignment was reported at default tool
  sensitivity;
- seed 1337 has zero exact 30-nt and 10-aa match coverage;
- seed 2027 has zero exact 30-nt coverage and mean 10-aa coverage `0.19%`, with a
  maximum of `9.71%`.

No reported alignment is not equivalent to zero biological similarity, but these
results do not suggest direct memorization.

## Decision

Unrestricted temperature-1.0 decoding is the correct base decoder for subsequent
comparisons, and it is substantially better than top-k 5/20. It is not reliable
enough for promotion: natural completion is inconsistent across seeds, hard-cap
rates remain high, and completed sequences tend to be short.

Proceed to the predeclared Phase 6 ablation:

1. termination head without replay;
2. termination head plus generated-prefix replay only if the head-only condition
   remains inadequate;
3. compare both against this unrestricted base decoder with matched prompts and
   seeds;
4. report natural-stop, early-stop, hard-cap, length-distribution, GC, PPL,
   runtime, and novelty effects.
