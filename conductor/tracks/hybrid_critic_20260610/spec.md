# Hybrid DNA-Protein Critic Benchmark Specification

## Overview
Implement a closed-loop bidirectional scoring filter that integrates the Multi-Task Critic (ProteinLM Classifiers) directly with the causal CodonLM generator. This moves beyond simple post-generation filtering by feeding back the critic's stability and classification logits to guide codon generation token-by-token.

## Requirements
- **Guided Generation Interface:** Modify the autoregressive generation loop to accept feedback from the ProteinLM classifiers at runtime.
- **Logit Blending:** Combine next-codon probabilities from CodonLM with corresponding stability and functionality log-probabilities computed from the partial protein sequence.
- **Metrics & Benchmarking:** Implement an automated evaluation run comparing standard ReD sampling with Hybrid Critic-Guided ReD sampling, recording sequence yield, validity, and compute overhead.

## Success Criteria
1. **ProteinCritic Stability Gain:** Blended generation yields $\ge 20\%$ relative increase in average predicted stability scores on 50 generated sequences compared to standard ReD.
2. **Grammar Preservation:** 100% of generated sequences retain correct coding structure (valid start codon, zero internal stop codons, valid terminal stop).
3. **Logit Blending Overhead:** Feedback loop adds $\le 1.0\times$ model evaluations (evaluating the critic alongside the generator) per step.
