# Targeted terminal weight-edit attempt (10 November 2026)

## Outcome

**Do not promote this checkpoint.** A direct, closed-form edit to the three biological stop-token rows gave a small validation perplexity improvement and raised the 300-token stop rate, but did not improve stop timing relative to held-out CDS lengths. It is not a demonstrated improvement in generated sequence quality.

## Edit

Source: `runs/corrected-codonlm-v1-batch64-lr-ablation-lr_1_5e4/checkpoints/best.pt`.

I extracted final-layer hidden states from the first 300 packed training rows. Positive examples were positions whose target was `TAA`, `TAG`, or `TGA` immediately followed by `<EOS_CDS>`; negatives were sampled sense-codon positions. There were 378 positives and 7,196 negatives. A ridge-regularized, class-balanced closed-form linear discriminant gave a direction in the 384-dimensional hidden space. I added that direction to the three stop-token rows of `head.weight`, choosing a scale on the first 100 validation rows (112 terminal examples) under a 0.003-nat/token NLL tolerance. No gradient optimizer or model retraining was used.

The selected scale was 0.25. Only `head.weight` changed, in 1,152 scalar entries. **The resulting row update norm is 2.391, versus source stop-row norms around 0.60–0.61.** The apparently small scale is therefore a large relative weight edit. This is a reason to constrain update norm in any future attempt.

## Validation and generation

| Metric | Source | Targeted edit | Earlier aggressive edit |
| --- | ---: | ---: | ---: |
| PPL, same first 50 validation rows | 48.995 | 48.973 | 49.012 |
| Biological stop, 100-token cap | 19/80 | 13/80 | 30/80 |
| Biological stop, 300-token cap | 44/80 | 50/80 | 63/80 |
| Hard cap, 300 tokens | 36/80 | 30/80 | 17/80 |
| Mean generated codons, 300-token cap | 211.0 | 208.7 | 161.2 |
| Mean GC, 300-token cap | 51.8% | 52.4% | 50.4% |
| Adjacent codon repeat rate, 300-token cap | 2.49% | 2.84% | 3.44% |

The targeted edit's 300-token stop-rate change is +7.5 percentage points, with a prompt-cluster bootstrap 95% interval from −5 to +20 points. Paired discordances were 15 gains and 9 losses (exact test p=0.307). Among the 50 trajectories whose source CDS continuation is no longer than 300 codons, stops within 75–125% of reference continuation length fell from **7 to 5**, while stops before half the reference length rose from **8 to 10**. The proposed edit did not solve the premature-stop problem.

The corrected ProteinCritic scored all 80 targeted continuations. Its mean predicted MegaScale ΔG was 1.734 kcal/mol, versus 1.738 for source generations; the paired difference was −0.004 kcal/mol (prompt-bootstrap interval −0.020 to +0.012). Family and function top-1 confidence changes also had intervals spanning zero. These are exploratory model scores, not experimental stability or function measurements.

## Interpretation and files

The hidden-state direction separated some teacher-forced terminal and nonterminal contexts, yet that separation did not translate into better free-running stop timing. The generated distribution moves off real CDS contexts, and the large row-norm change makes the edit hard to justify causally. The earlier conservative head scaling remains better on validation PPL, while neither edit has demonstrated better generation quality.

- Derivation: `scripts/derive_weight_space_terminal_edit.py`.
- Experimental checkpoint and scale scan: `runs/corrected-targeted-terminal-weight-edit-seed1337/checkpoints/`.
- Paired trajectories: `runs/_summary/weight_space_targeted_eval_100/` and `runs/_summary/weight_space_targeted_eval_300/`.
- Matched source and aggressive trajectories: `runs/_summary/weight_space_paired_eval/` and `runs/_summary/weight_space_paired_eval_300/`.
