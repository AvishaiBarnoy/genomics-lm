# Bidirectional Backbone & Attention-Pooling for MultiTask ProteinCritic Spec

## Overview
Integrate bidirectional attention into the ProteinCritic backbone, and replace average pooling with learnable attention-based pooling for active-site focus and saliency visualization.

## Requirements
- Introduce bidirectional attention mapping to replace standard causal masks.
- Implement a learnable attention pooling layer instead of global average pooling.

## Success Criteria
1. **Stability Classifier Parity:** The attention-pooled bidirectional critic converges to a validation accuracy $\ge 77\%$ on thermodynamic stability, equal to or better than the average-pooling baseline.
2. **Attention Saliency Contrast:** Active-site codons (from catalytic residues in reference databases) receive $\ge 2\times$ higher attention weights on average compared to non-catalytic structural loops.
