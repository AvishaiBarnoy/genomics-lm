# Architectural Upgrades (RoPE & SwiGLU) for CodonLM Spec

## Overview
Upgrade the CodonLM backbone with Rotary Position Embeddings (RoPE) and SwiGLU Feed-Forward Networks.

## Requirements
- Introduce RoPE to replace absolute positional embeddings.
- Upgrade standard feed-forward networks (FFN) to SwiGLU.
- Verify mathematical compatibility and throughput.

## Success Criteria
1. **Relative Perplexity Reduction:** Upgraded model (with RoPE and SwiGLU) achieves a lower training perplexity at the same token step compared to standard positional/FFN configurations (e.g., relative reduction $\ge 5\%$).
2. **Probing Task Gains:** R² score on structural shape regression probes increases by $\ge 8\%$, showing that RoPE improves continuous spatial/biophysical representations.
