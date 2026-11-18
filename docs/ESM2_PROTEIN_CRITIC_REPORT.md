# External Protein Language Model Critic (ESM-2 8M) Benchmark Report

**Date:** 2026-11-18  
**Status:** Completed & Validated (Pass)  
**Track:** [conductor/tracks/esm2_protein_critic_20261116/plan.md](file:///Users/User/github/genomics-lm/conductor/tracks/esm2_protein_critic_20261116/plan.md)  
**Implementation Modules:**
- [src/eval/protein_critic_esm.py](file:///Users/User/github/genomics-lm/src/eval/protein_critic_esm.py)
- [scripts/eval_esm2_critic.py](file:///Users/User/github/genomics-lm/scripts/eval_esm2_critic.py)
- [tests/test_protein_critic_esm.py](file:///Users/User/github/genomics-lm/tests/test_protein_critic_esm.py)

---

## 1. Executive Summary

This benchmark established the **Generator $\to$ Translation $\to$ External Critic** evaluation interface by integrating a frozen, pretrained **ESM-2 (8M parameter, `esm2_t6_8M_UR50D`)** protein language model as an objective sequence plausibility critic.

We evaluated 32 *de novo* generated coding sequences against 32 length-matched real validation proteins from the held-out genome validation split. For each sequence, we calculated the mean masked-residue log-likelihood across 16 randomly selected positions.

### Key Findings
1. **Finite Score Invariant**: All 64 protein sequences received finite, well-conditioned scores (ranging between $-3.6$ and $-1.8$ nats).
2. **Natural Plausibility Advantage**: Natural validation proteins received significantly higher plausibility scores from ESM-2 than generated candidates:
   $$\text{Median}_{\text{real}} = -2.6142 \quad\text{vs.}\quad \text{Median}_{\text{generated}} = -2.8613 \quad(\Delta\text{Median} = +0.2471)$$
   The difference is statistically significant under a one-sided Mann-Whitney U test ($U = 770.0, p = 2.73 \times 10^{-4}$).
3. **Decoupled Biological Evaluation**: Confirms that a lightweight 8M model (~30 MB weights) running on local hardware provides an effective, non-circular biological plausibility filter for generated sequences.

---

## 2. Experimental Setup

- **Generator Checkpoint:** `runs/corrected-codonlm-v1-batch64-lr15-seed2027` ([checkpoints/best.pt](file:///Users/User/github/genomics-lm/runs/corrected-codonlm-v1-batch64-lr15-seed2027/checkpoints/best.pt))
- **Generator Generation Protocol:** 32 sequences, max 128 codons, seed 42, temperature 1.0, unconstrained sampling (`top_k=0`).
- **Translation:** Standard bacterial code (NCBI Table 11) via [src/eval/translation.py](file:///Users/User/github/genomics-lm/src/eval/translation.py).
- **Validation Dataset:** [data/processed/corrected/corrected-codonlm-v1/genome/manifest.json](file:///Users/User/github/genomics-lm/data/processed/corrected/corrected-codonlm-v1/genome/manifest.json) (length-matched to generated proteins).
- **External Critic:** Pretrained `esm2_t6_8M_UR50D` (Meta AI, UniRef50-trained).
- **Masked Log-Likelihood Formulation:**
  For residue position $i_k$, replace input token with `<mask>` (token 32), compute forward pass, and evaluate log-softmax probability of the ground-truth residue $x_{i_k}$:
  $$\text{Score} = \frac{1}{K} \sum_{k=1}^{K} \ln P(x_{i_k} = \text{target}_{i_k} \mid x_{\setminus i_k}) \quad (K = 16)$$

---

## 3. Results Summary

| Metric | Real Validation Proteins ($N=32$) | Generated Proteins ($N=32$) | Difference / Test |
|---|---|---|---|
| **Median Score** | **-2.6142** nats | **-2.8613** nats | **$\Delta\text{Median} = +0.2471$** |
| **Mean Score** | $-2.5273 \pm 0.3986$ | $-2.8307 \pm 0.2009$ | $\Delta\text{Mean} = +0.3034$ |
| **Perplexity ($\exp(-\text{Score})$)** | $12.52$ | $16.96$ | Real perplexity is $26.2\%$ lower |
| **All Scores Finite** | True | True | Pass |
| **Mann-Whitney U Test** | — | — | $U = 770.0$, $p = 2.73 \times 10^{-4}$ |
| **Success Check** | — | — | **PASS** |

---

## 4. Reproducibility

To re-run the benchmark on any checkpoint:

```bash
python -m scripts.eval_esm2_critic \
    --run_dir runs/corrected-codonlm-v1-batch64-lr15-seed2027 \
    --num_seqs 32 \
    --max_codons 128 \
    --num_positions 16 \
    --seed 42 \
    --device cpu
```

Output report: [runs/corrected-codonlm-v1-batch64-lr15-seed2027/scores/esm2_critic_benchmark.json](file:///Users/User/github/genomics-lm/runs/corrected-codonlm-v1-batch64-lr15-seed2027/scores/esm2_critic_benchmark.json)
