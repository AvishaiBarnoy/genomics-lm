# NoProp Integration Specification

## Overview
Implement NoProp (No Propagation) training for the Genomics-LM project to overcome memory limitations on Apple M2 hardware when scaling depth and context lengths.

## Requirements
- Introduce decoupled block architectures (`NoPropBlock`, `NoPropTinyGPT`) that parallel `TinyGPT`.
- Implement a custom training loop (`train_noprop.py`) utilizing local objectives (e.g., MSE denoising on target embeddings) instead of global backpropagation.
- Preserve the existing `TinyGPT` global backprop flow as a functional fallback.
- Provide PyTorch unit tests verifying gradient isolation (gradients do not flow between blocks) and constant memory profiling across depth.

## Success Criteria
1. **Gradient Isolation:** 100% confirmation via PyTorch state-checks that gradients are isolated (a `loss.backward()` in block $i$ updates weights in block $i$ only, leaving block $i-1$ unchanged).
2. **Memory Scaling Parity:** Peak GPU/MPS memory consumption scales as $O(1)$ (constant) with model depth, demonstrating $\le 10\%$ increase in memory overhead when scaling from 6 to 12 layers (compared to $\ge 90\%$ increase in standard backpropagation).
3. **Validation Convergence:** NoProp training converges to $\ge 85\%$ of the next-token prediction accuracy of standard backprop training on a matched toy sequence dataset.