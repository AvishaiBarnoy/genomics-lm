# Hybrid DNA-Protein Critic Benchmark Specification

## Overview
Implement a closed-loop bidirectional scoring filter that integrates the Multi-Task Critic (ProteinLM Classifiers) directly with the causal CodonLM generator. This moves beyond simple post-generation filtering by feeding back the critic's stability and classification logits to guide codon generation token-by-token.

## Requirements
- **Guided Generation Interface:** Modify the autoregressive generation loop to accept feedback from the ProteinLM classifiers at runtime.
- **Logit Blending:** Combine next-codon probabilities from CodonLM with corresponding stability and functionality log-probabilities computed from the partial protein sequence.
- **Metrics & Benchmarking:** Implement an automated evaluation run comparing standard ReD sampling with Hybrid Critic-Guided ReD sampling, recording sequence yield, validity, and compute overhead.
