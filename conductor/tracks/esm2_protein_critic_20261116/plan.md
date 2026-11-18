# ESM-2 (8M) External Protein Critic Plan

## Overview
Integrate a frozen external protein language model (ESM-2 8M, `esm2_t6_8M_UR50D`) as an objective plausibility critic for translated CodonLM generations.

## Methodology
- **External Critic**: ESM-2 8M parameter model evaluated in inference mode on Apple MPS/CPU.
- **Generator**: 32 sequences generated from trained CodonLM checkpoint (`runs/corrected-codonlm-v1-batch64-lr15-seed2027`), max length 128 codons, seed 42, batch size 1.
- **Translation**: Translate generated CDS into protein sequences using NCBI Table 11 ([src/eval/translation.py](file:///Users/User/github/genomics-lm/src/eval/translation.py)).
- **Length-Matched Real Control**: 32 validation proteins length-matched to the generated set from the held-out genome validation split.
- **Scoring**: For each protein, compute mean masked-residue log-likelihood across 16 randomly selected residue positions:
  $$\text{Score} = \frac{1}{K} \sum_{k=1}^{K} \ln P(x_{i_k} = \text{target}_{i_k} \mid x_{\setminus i_k})$$
- **Success Criteria**:
  1. All 64 proteins receive finite scores.
  2. The real-protein set has a higher median score than the generated set ($\text{Median}_{\text{real}} > \text{Median}_{\text{generated}}$).

## Tasks
- [x] Install `fair-esm` and cache pretrained `esm2_t6_8M_UR50D` weights.
- [x] Implement `ESM2ProteinCritic` in [src/eval/protein_critic_esm.py](file:///Users/User/github/genomics-lm/src/eval/protein_critic_esm.py).
- [x] Implement CLI benchmark in [scripts/eval_esm2_critic.py](file:///Users/User/github/genomics-lm/scripts/eval_esm2_critic.py).
- [x] Add unit tests in [tests/test_protein_critic_esm.py](file:///Users/User/github/genomics-lm/tests/test_protein_critic_esm.py) (6/6 pass).
- [x] Run benchmark on 32 generated vs 32 real validation proteins.
- [x] Document benchmark results and findings.

## Benchmark Results (`runs/corrected-codonlm-v1-batch64-lr15-seed2027`)

| Metric | Real Validation Proteins | Generated Proteins | Delta (Real - Generated) |
|---|---|---|---|
| **Count** | 32 | 32 | — |
| **Evaluated Positions** | 16 | 16 | — |
| **All Scores Finite** | True | True | Pass |
| **Median Score** | **-2.6142** nats | **-2.8613** nats | **$+0.2471$** |
| **Mean Score** | $-2.5273 \pm 0.3986$ | $-2.8307 \pm 0.2009$ | $+0.3034$ |
| **Perplexity** | 12.52 | 16.96 | — |
| **Mann-Whitney U** | — | — | $U=770.0, p=2.73 \times 10^{-4}$ |
| **Verdict** | — | — | **PASS** |

