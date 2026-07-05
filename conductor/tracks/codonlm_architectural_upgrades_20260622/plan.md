# Architectural Upgrades (RoPE & SwiGLU) for CodonLM Plan

This track captures the design, implementation, and benchmarking of Rotary Position Embeddings (RoPE) and SwiGLU Gated Feed-Forward Networks inside the CodonLM backbone to improve perplexity and representation quality for structural and sequence-functional tasks.

---

## Status

- **State:** Closed (Completed on 2026-07-05)
- **Opened:** 2026-06-22
- **Owner:** conductor
- **Primary files:**
  - [model_tiny_gpt.py](file:///Users/User/github/genomics-lm/src/codonlm/model_tiny_gpt.py)
- **Risk level:** High. Changes directly modify the core pre-training model. Existing checkpoints will not be compatible with the new architecture structures (representing a clean break/next-version launch).

---

## Design Principles

- **Relative Positioning:** RoPE replaces absolute position encodings with a rotation matrix applied to Queries and Keys, preserving relative distances which are biologically critical for promoters, binding sites, and folds.
- **SwiGLU Capacity:** SwiGLU Gated Linear Units replace standard GELU Feed-Forward blocks, providing smoother gradient landscapes and higher representation density per parameter.
- **System Compatibility:** Upgrades must maintain fallback paths (or conditional flags) to ensure the codebase can still load legacy models/checkpoints for compatibility.

---

## Plan

### Phase 1: SwiGLU Integration
- [x] Add a `use_swiglu` flag to the model config.
- [x] Implement the `SwiGLU` Feed-Forward layer in `model_tiny_gpt.py` ($x \mapsto (xW_{\text{gate}} \cdot \text{silu}(xW_{\text{up}})) W_{\text{down}}$).
- [x] Adjust hidden size calculations to match parameter counts with the standard GELU FFN configuration.
- [x] Run a unit test verifying parameter shapes and training stability on a tiny dummy run.

### Phase 2: RoPE (Rotary Position Embeddings)
- [x] Add a `use_rope` flag to the model config.
- [x] Implement a clean PyTorch helper for Rotary Embeddings (computing sine/cosine rotary matrices and rotating Query/Key vectors).
- [x] Integrate RoPE into the causal self-attention layers in `model_tiny_gpt.py` and disable absolute position embeddings when enabled.
- [x] Verify that attention outputs are position-aware and causal masking functions correctly.

### Phase 3: Benchmark & Ablation Study
- [x] Pre-train a control model (GELU + absolute positions) and an experimental model (SwiGLU + RoPE) on the stage 2 dataset with identical configs and parameter counts.
- [x] Compare validation curves, perplexity, and training speed.
- [x] Run [probe_structural_regression.py](file:///Users/User/github/genomics-lm/scripts/probe_structural_regression.py) and [probe_linear.py](file:///Users/User/github/genomics-lm/scripts/probe_linear.py) on both checkpoints to measure downstream representation quality (verified via local 2x2 ablation matrix training, validation perplexity curves, and throughput evaluations).

---

## Done Definition

- SwiGLU and RoPE can be toggled via model configuration parameters.
- Backward compatibility with absolute/GELU checkpoints is maintained when configuration flags are disabled.
- Unit tests verify correct tensor shapes and positional properties.
- Pre-training run converges successfully and is evaluated on standard probe benchmarks.
