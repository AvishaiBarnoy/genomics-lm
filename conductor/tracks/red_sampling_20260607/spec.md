# Spec: Reset-and-Discard (ReD) Sampling

## Problem Statement
Current inference benchmarks for CodonLM (e.g., `2026-06-05_stage2.5_6L4H_d256_e10`) show a 0% termination rate. Models hit the hard cap without generating a stop codon. Standard sampling (Solve-to-completion) is inefficient when failure probabilities follow a power law, as hard instances consume disproportionate compute.

## Solution
Implement the **Reset-and-Discard (ReD)** strategy as described in [Meir et al. (2026), "More Bang for the Buck"]. ReD optimizes "coverage@cost" by resetting failed attempts early and reallocating budget to fresh, independent trials across a pool of tasks.

## Objectives
1.  **Efficient Coverage:** Maximize the number of genomic prefixes for which a valid terminal stop codon is found within a fixed token budget.
2.  **Benchmark Gains:** Quantify the reduction in tokens required to reach a target coverage level compared to standard sequential sampling.
3.  **Modular Inference:** Provide a ReD-aware generator that can be used in other scripts.

## Success Criteria
-   `src/codonlm/generate.py` includes a `generate_cds_red` function.
-   `scripts/benchmark_red.py` demonstrates a higher coverage@cost compared to the standard policy.
-   Verification on a failing run shows non-zero coverage within a budget that previously yielded 0%.
