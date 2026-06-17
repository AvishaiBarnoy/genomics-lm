# Long-Range CodonLM Objectives Plan

## Status

In progress. The first implementation pass adds config-gated multi-offset
future-token losses, next-token-preserving metrics, and whole-gene pack audits.

## Tasks

- [x] Add a config-gated multi-offset LM loss.
- [x] Add tests for offset target creation and end-of-sequence masking.
- [x] Keep next-token perplexity separate from auxiliary-loss validation.
- [x] Add whole-gene/truncation audit metadata for dynamic packs.
- [x] Add a smoke config with offsets `+4`, `+8`, `+16`, `+32`.
- [ ] Run a tiny smoke experiment with the long-range objective enabled.
- [ ] Compare validation next-token perplexity against standard training.
- [ ] Rescore generated libraries with calibrated ProteinCritic selection rules.
- [ ] Add generated-prefix replay or denoising corruption.
- [ ] Build an offline hard-negative dataset from generated and corrupted ORFs.
- [ ] Run a conservative d384-vs-d512 capacity ablation only after objective/data metrics improve.

## Biology-Informed Priors

- Helical local structure often creates useful residue relationships around
  nearby offsets such as `i -> i+3/i+4`.
- Medium-range sequence constraints matter for motifs, beta-strand pairing,
  domain cores, and topology, but they are weakly supervised by pure next-token
  loss.
- The multi-offset objective is therefore a diagnostic training pressure, not a
  complete structural objective.

## Initial Experiment

Use `configs/long_range_offsets_smoke.yaml` from a backed-up Stage 2.6 checkpoint.
Accept the objective only if:

- next-token validation perplexity regresses by no more than 2%;
- generation does not collapse into short peptides or non-terminating outputs;
- calibrated ProteinCritic top-fraction enrichment improves for at least one
  structural/useful label without broad degradation.
