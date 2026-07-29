# Corrected Base-Model Generation Evaluation

## Question

Can the corrected basic CodonLM continue a start-codon prefix into a plausible CDS
without termination guidance, critic guidance, replay, or forced terminal tokens?

This is a generation diagnostic, not evidence of protein function or fitness.

## Protocol

- Frozen genome-held-out test source records only.
- 50 prompts, balanced 25/25 across the two held-out genomes.
- One codon of natural prefix and one sample per held-out CDS.
- Paired raw-vocabulary and CDS-token-constrained sampling.
- Temperature `0.8`, top-k `5`, maximum 300 new tokens.
- The CDS-constrained decoder stops at its declared 256-codon target if no natural
  stop occurs. That length is imposed by the evaluator and is not learned.
- Corrected checkpoints seeds 1337 and 2027.
- Exact 10- and 20-codon training-match coverage indexed from ten million packed
  training tokens.

## Results

| Seed | Protocol | Natural stop | Hard cap | Mean total codons | Unique | Mean GC | Codon JS vs held-out sources |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1337 | Raw vocabulary | 0/50 | 50/50 | 301 | 100% | 74.4% | 0.294 |
| 1337 | CDS constrained | 0/50 | 0/50 | 257 | 100% | 74.3% | 0.292 |
| 2027 | Raw vocabulary | 0/50 | 50/50 | 301 | 100% | 76.2% | 0.305 |
| 2027 | CDS constrained | 0/50 | 0/50 | 257 | 100% | 76.3% | 0.303 |

The selected held-out source CDSs have mean GC `52.9%`. No generated sequence
contains an internal or terminal biological stop. Indexed exact training-match
coverage is zero at both 10 and 20 codons for every generated sequence.

The CDS-constrained hard-cap rate is zero only because that protocol deliberately
returns at its 256-codon target. It does not represent natural completion.

## Interpretation

The corrected base model passes the intrinsic PPL gate but fails the natural
generation gate. Teacher-forced next-token prediction and free-running generation
are different regimes: small distributional errors compound after the model begins
conditioning on its own samples.

The replicated failure has two components:

1. Neither checkpoint places a biological stop or EOS in 50 near-de-novo samples.
2. Both checkpoints drift toward an unusually high-GC codon distribution.

The samples are diverse and show no indexed long exact matches, so the observed
failure is not simple sequence duplication or sample collapse.

## Remaining Audit

The exhaustive nucleotide/protein nearest-neighbor alignment against all 74,600
training CDSs remains blocked on the 8 GB host. MMseqs2 could not allocate the
nucleotide target database even with 512 MB split limits. This must be completed
with explicit training-database chunking or on a higher-memory host before the
memorization audit is considered complete.

## Decision

Do not promote the basic model for unguided gene generation. The result motivates
the predeclared termination/replay phase and a separate decoding/composition
diagnostic. Syntax constraints may be reported as an inference intervention, but
they do not repair the model's missing length prior or GC drift.
