# Structured Protein Generation Closeout Report

**Track:** `structured_generation_20260616`
**Closed:** 2026-06-16
**Conclusion:** Sampling-time structure heuristics improved ProteinCritic scores, but did not improve ESMFold pLDDT. Future pLDDT gains need a structural training signal.

## Starting Point

The preceding Generative Design Loop produced a diverse terminated library, but ESMFold confidence stayed low:

| Baseline observation | Result |
|---|---:|
| Generated sequences | 50 |
| Termination | 50/50 |
| Mean pairwise identity | ~9.2% |
| ESMFold pLDDT range for tested top sequences | ~0.4-0.6 |
| Interpretation | diverse but likely disordered/de novo sequences |

The working hypothesis for this track was that sampling-time constraints could bias the generator toward better-folding proteins without retraining.

## Implemented Options

### T1a: Critic-Guided ReD

Implemented `--min_stability` and `--max_stability_attempts` in `scripts/generative_design_loop.py`.

Mechanism:
- Generate with ReD until a sequence terminates and meets minimum AA length.
- Score the AA sequence with the MultiTask ProteinCritic.
- Reject candidates whose `stability_prob` is below threshold.
- Keep the best candidate as a fallback if all outer attempts fail.

Expected effect before implementation:
- Increase the fraction of accepted sequences that look stable to the critic.
- Potentially raise ESMFold pLDDT if critic stability aligned with foldability.

Observed effect:

| Run | n | stability_mean | Change vs baseline |
|---|---:|---:|---:|
| Baseline design loop | 50 | 0.608 | - |
| T1a+T1c smoke | 5 | 0.689 | +13.4% |
| T1a+T1c full | 50 | 0.691 | +13.6% |

This achieved the critic-score objective, but not the structural objective.

### T1b: Family-Targeted Generation

Implemented `--target_family_idx` and `--min_family_conf`.

Mechanism:
- Reuse the same outer rejection loop as T1a.
- Accept only candidates whose `family_top1` matches the requested class and whose confidence exceeds the configured threshold.

Observed limitation:
- In the T1a full run, all 50 accepted sequences mapped to the same weak top family (`family_top1=297`) with low confidence (~0.085).
- This means the family head was not confident enough for strong fold-family steering on these de novo outputs.

### T1c: Temperature Annealing

Implemented `--anneal_temp`.

Mechanism:
- Start at the requested sampling temperature.
- After 50 codons, linearly anneal toward `0.7 * temperature`.

Expected effect before implementation:
- Preserve N-terminal exploration while making longer generated cores more conservative.

Observed effect:
- The T1a+T1c full run remained highly diverse and terminated 50/50.
- It did not create a measurable pLDDT gain in the tested top sequences.

### T2b: Top-p / Nucleus Sampling

Implemented `--top_p`.

Mechanism:
- When `top_p > 0`, sorted logits are truncated by cumulative probability mass.
- This overrides hard top-k sampling.

Expected effect before implementation:
- Avoid the brittle hard cutoff of top-k and improve sampling calibration.

Observed status:
- Implemented and available for future experiments.
- No completed full pLDDT benchmark is recorded for top-p alone in this track, so it should not be claimed as a structural improvement.

### T2a: Structured-Prefix Experiment Harness

Implemented `scripts/structured_prefix_experiment.py`.

Mechanism:
- Hardcodes prompts for three known structured enzyme families:
  - DHFR/FolA-like
  - TEM-1 beta-lactamase-like
  - TPI/TIM-barrel-like
- Generates CodonLM continuations from each prefix.
- Scores every continuation with the same MultiTask ProteinCritic.
- Optionally submits top candidates to ESMFold via `--esm_fold_top`.

Expected effect before implementation:
- A fold-family prefix may keep generation near a more foldable region of sequence space.

Observed effect after running 10 continuations per prefix and submitting all 30 to ESMFold:

| Metric | Result |
|---|---:|
| Prefixes | 3 |
| Generated continuations | 30 |
| Terminated within cap | 0/30 |
| Mean critic stability | 0.571 |
| Mean ESMFold pLDDT | 0.317 |
| Median ESMFold pLDDT | 0.320 |
| Min / Max ESMFold pLDDT | 0.266 / 0.383 |
| pLDDT > 0.7 | 0/30 |

Structured prefixes did not improve fold confidence. The continuations were long and diverse, but every ESMFold prediction remained low-confidence.

## Full T1a+T1c Run

Source artifact: `outputs/reports/structured_gen_t1a/`

| Metric | Result |
|---|---:|
| Generated sequences | 50 |
| Properly terminated | 50/50 |
| Avg attempts per sequence | 10.92 |
| Mean AA length | 106.3 +/- 62.7 |
| Mean GC content | 63.6% |
| Mean stability probability | 0.691 |
| P(stable) > 0.7 | 1/50 |
| Mean Pfam top-1 confidence | 0.0859 |
| Mean EC top-1 confidence | 0.1630 |
| Mean pairwise AA identity | 9.7% |
| 3-mer AA k-mer coverage | 28.35% |

The library remained diverse and properly terminated, but the critic heads were low-confidence apart from a narrow stability score shift.

## ESMFold Finding

Top candidates from both critic-guided generation and structured-prefix generation did not improve structurally:

| Run | Candidate | Critic stability | ESMFold pLDDT |
|---|---:|---:|---:|
| Baseline | seq 1 | - | 0.41 |
| Baseline | seq 2 | - | 0.50 |
| Baseline | seq 3 | - | 0.60 |
| T1a+T1c | seq 17 | 0.754 | 0.41 |
| T1a+T1c | seq 38 | 0.689 | 0.288 |
| T1a+T1c | seq 20 | 0.689 | 0.471 |
| T2a structured prefix | best of 30 | 0.574 | 0.383 |

The best critic-scored sequence and best structured-prefix sequence did not fold better than the baseline top candidates. The stability head learned useful natural-sequence discrimination, and structured prefixes can bias local context, but neither is a reliable pLDDT proxy for de novo generated proteins.

## Interpretation

The implemented sampling controls affected acceptance statistics but not the actual structural confidence target.

Practical reading:
- ReD solves validity/termination.
- Critic-guided ReD improves the critic score it is optimizing.
- Family filtering is currently weak because family confidence is low on generated sequences.
- Annealing and top-p are useful sampling controls, but they do not inject structural supervision.
- Structured family prefixes alone are insufficient: 30/30 prefix continuations remained below pLDDT 0.7.
- ESMFold pLDDT is a separate target that requires either structural data in training or direct reward feedback.

Scientific reading:
- This track falsified the initial cheap-fix hypothesis.
- A natural-sequence critic is not interchangeable with a structure oracle.
- The result is useful enough to report: critic-guided selection can make generated proteins look more natural to a classifier while remaining low-confidence under a structure model.

## Recommendation

Close this track as complete with a negative result. Open a follow-up track only if we are ready to add a true structural signal:

1. **PDB-filtered fine-tuning:** filter training CDS/proteins to known-structure proteins and fine-tune CodonLM for one epoch.
2. **Direct ESMFold reward:** use pLDDT as a scalar reward with KL regularization against the current CodonLM.
3. **Secondary-structure conditioning:** condition generation with helix/sheet/coil tags after building a PDB/DSSP-derived dataset.

The most practical next step is PDB-filtered fine-tuning. It is simpler and less brittle than reinforcement learning, and it addresses the data-bias root cause directly.
