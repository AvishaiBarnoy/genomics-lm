# NoProp Algorithm Integration Plan

**Status:** Prototype implemented; not trained/evaluated.

**Update (2026-06-16):** We added the NoProp architecture, a dedicated training
script, a toy config, and unit tests. We have **not** run a substantive NoProp
training experiment, benchmarked memory scaling, or integrated NoProp
checkpoints into the standard evaluation/generation tools. Keep this track open
until those validation steps are done.

## Background & Motivation
The current models (`TinyGPT` for CodonLM, `ProteinLM`) are trained using standard global backpropagation. As sequence lengths increase (especially for the upcoming Nucleotide-Level models) and model sizes expand, we are hitting memory bottlenecks on the 8GB Apple M2 hardware. 
NoProp algorithms decouple the network into independent denoising layers, eliminating the need to store activations for the full forward and backward pass. This provides constant memory scaling with respect to network depth.

## Scope & Impact
This track focuses on creating a secondary NoProp-compatible architecture and training loop for CodonLM as a proof-of-concept.
- **Affected Components**: 
  - `src/codonlm/model_tiny_gpt.py`: Introduce `NoPropBlock` and `NoPropTinyGPT`.
  - `src/codonlm/train_codon_lm.py` (or a new `train_noprop.py`): Implement local block-wise optimizers and independent layer denoising steps.
- **Impact**: It will decouple model layers, allowing deeper architectures on 8GB RAM at the cost of changing the learning paradigm. Standard TinyGPT remains untouched as a reliable fallback.

## Proposed Solution
1. **Architecture Modification**: Build a `NoPropTinyGPT` variant where each `NoPropBlock` predicts a clean target embedding from a noisy label embedding, conditioned on the sequence context. 
2. **Local Objectives**: In the training loop, instead of a global `loss.backward()` over the full model, iterate over blocks. Add noise to target embeddings, pass them into the block alongside the hidden state from the previous block, compute a local MSE loss against the clean target embedding, and step the local optimizer.
3. **Inference**: During inference, standard autoregressive generation can still apply, using the output of the final layer as the token logits, or using a specialized decoding strategy dependent on the chosen NoProp variant.

## Alternatives Considered
- **Gradient Accumulation & Checkpointing**: Already in use (Stage 2.5), but still hits fundamental limits as we increase sequence length to 1.5kb+ and add layers.
- **LoRA / QLoRA**: Reduces parameter optimizer states but still requires full forward/backward passes for the backbone.
- **Upgrading Hardware**: Not an immediate option; we must maximize the M2 constraints.

## Implementation Steps
1. **Phase 1: Architecture (NoProp Modules)** - Done
   - [x] Add `NoPropBlock` to `src/codonlm/model_tiny_gpt.py`.
   - [x] Add `NoPropTinyGPT` that initializes multiple `NoPropBlock`s and a shared target embedding matrix.
2. **Phase 2: Training Loop** - Prototype done, not validated
   - [x] Create `src/codonlm/train_noprop.py` to initialize a list of optimizers (one per block).
   - [x] Implement the local forward-backward pass:
     - Generate Gaussian noise.
     - Compute noisy target embeddings.
     - For each block, predict clean embeddings, compute local MSE loss, and run `backward()` locally.
   - [ ] Run a real NoProp training experiment beyond the toy/smoke path.
   - [ ] Compare validation loss, throughput, and peak memory against standard TinyGPT backprop.
3. **Phase 3: Validation & Inference** - Open
   - [ ] Ensure `eval_perplexity.py` and generation scripts (`sample.py` / `inference_playground.py`) can load and query `NoPropTinyGPT`.
   - [ ] Decide whether NoProp remains an experimental separate trainer or becomes selectable from the main training config.

## Verification & Testing
- [x] Create `tests/test_noprop.py` to assert that local gradients only affect the current block.
- [ ] Add memory-scaling profiling across increasing `n_layer`.
- [ ] Run a minimal training loop on a small subset of CDS data to verify loss decreases.

## Current Artifacts
- `src/codonlm/model_tiny_gpt.py`: `NoPropBlock`, `NoPropTinyGPT`
- `src/codonlm/train_noprop.py`: standalone NoProp trainer
- `configs/noprop_toy.yaml`: toy configuration
- `tests/test_noprop.py`: initialization, gradient isolation, and forward-shape tests
- `runs/2026-06-14_noprop_2L2H_d64_e1`: smoke artifact only; not enough to claim training success

## Migration & Rollback
- The existing `TinyGPT` and `train_codon_lm.py` scripts will remain as the default. The NoProp integration will be opt-in via a config flag (e.g., `use_noprop: true`) or a dedicated script (`train_noprop.py`).
- Rollback simply entails continuing to use the standard `TinyGPT` configuration.
