# CodonLM Trainer Refactor Spec

## Overview
This track governs the refactoring of `src/codonlm/train_codon_lm.py` into smaller, modular, and highly testable packages to improve maintainability, validation speed, and correctness in our training pipelines.

## Requirements
- Decouple training setup, dataloader packaging, execution runtime loop, and state checkpointing/resumption.
- Maintain absolute backward compatibility with current configs, CLI arguments, and checkpoints.
- Ensure all modules are fully covered by standard unit tests.

## Success Criteria
1. **Refactoring Parity:** The refactored training loop yields identical training trajectories (equal losses at identical steps/epochs up to float32 precision) to the original unified script on a matched configuration.
2. **Checkpoint Compatibility:** Pre-existing model checkpoints (Stage 2.5/2.6) load successfully into the refactored code without architecture errors.
