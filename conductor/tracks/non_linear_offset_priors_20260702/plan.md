# Non-Linear Offset Priors & Backbone Freezing Plan

## Status
Completed. Successfully integrated non-linear MLP projection heads, parameter-efficient backbone freezing, and evaluated the resulting model.

## Objectives
1. **Backbone Freezing**: Freeze `transformer` and `head` parameters during auxiliary offset training. This mathematically guarantees next-token perplexity remains identical to the pretrained Stage 2.6 baseline.
2. **Non-Linear MLP Heads**: Replace the single linear projections with 2-layer MLPs (`Linear -> GeLU -> Linear`) for offsets `[2, 4, 8, 16, 32]`.
3. **Identity Warmup**: Initialize MLP heads to behave as near-identity mappings at step 0 to preserve pretrained representations.
4. **Longer Training**: Train for 5–10 epochs to allow the MLP projections to converge fully.
5. **Matched Evaluation**: Score generated sequences using the MultiTask ProteinCritic stability/classification metrics.

## Tasks
- [x] Implement MLP heads with identity initialization in `TinyGPT`.
- [x] Add unit tests verifying MLP identity behaviour.
- [x] Add config-gated backbone freezing to `train_codon_lm.py`.
- [x] Add unit tests verifying backbone freezing works correctly during training.
- [x] Create `configs/separate_heads_mlp_frozen.yaml`.
- [x] Launch training run `separate_heads_mlp_frozen`.
- [x] Run evaluations with balanced prior weights.
