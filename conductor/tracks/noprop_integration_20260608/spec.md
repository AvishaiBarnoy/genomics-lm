# NoProp Integration Specification

## Overview
Implement NoProp (No Propagation) training for the Genomics-LM project to overcome memory limitations on Apple M2 hardware when scaling depth and context lengths.

## Requirements
- Introduce decoupled block architectures (`NoPropBlock`, `NoPropTinyGPT`) that parallel `TinyGPT`.
- Implement a custom training loop (`train_noprop.py`) utilizing local objectives (e.g., MSE denoising on target embeddings) instead of global backpropagation.
- Preserve the existing `TinyGPT` global backprop flow as a functional fallback.
- Provide PyTorch unit tests verifying gradient isolation (gradients do not flow between blocks) and constant memory profiling across depth.