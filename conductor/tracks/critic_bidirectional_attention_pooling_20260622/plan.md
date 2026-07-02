# Bidirectional Backbone & Attention-Pooling for MultiTask ProteinCritic Plan

This track captures the design and implementation of structural upgrades to the ProteinCritic. The goal is to move from a causally-masked average-pooled backbone to a bidirectional attention backbone with learnable attention pooling to focus features on functional motifs.

---

## Status

- **State:** Open
- **Opened:** 2026-06-22
- **Owner:** conductor
- **Primary files:**
  - [models_multi.py](file:///Users/User/github/genomics-lm/src/protein_lm/models_multi.py)
  - [train_multi_task.py](file:///Users/User/github/genomics-lm/src/protein_lm/train_multi_task.py)
- **Risk level:** Low to Medium. Changes are focused on the ProteinCritic classifier path; main CodonLM generative pre-training is unaffected.

---

## Design Principles

- **Bidirectional Attention:** Since the critic evaluates full sequences for stability, Pfam class, and EC function, causality is not required. Bidirectional attention allows the model to capture symmetric 3D folding interactions.
- **Attention-Based Pooling:** Replace global average pooling with a learnable query-key attention module, allowing the critic to learn which residues (e.g., active sites, structural loops) are predictive for family and function.
- **Shared Latent Layer Bottleneck:** Project the pooled representations through a shared non-linear projection layer before routing to individual classifier heads. This forces joint representation learning of biological structure, function, and stability to act as a regularizer.
- **Interpretability:** Retain attention pooling weights as saliency maps for direct visualization of what the critic "looks at."

---

## Plan

### Phase 1: Bidirectional Attention Configuration
- [ ] Add a config flag (e.g., `bidirectional: true` or `causal: false`) to `ProteinClassifierConfig`.
- [ ] Modify the self-attention blocks in `ProteinConditionalTransformer` to optionally disable causal masking when running under the classification backbone.
- [ ] Run a sanity check or unit test to verify that hidden states depend on both future and past tokens.

### Phase 2: Attention-Pooling Module
- [ ] Implement an `AttentionPooling` module in `models_multi.py`.
- [ ] Add a learnable Query parameter $q \in \mathbb{R}^{d_{\text{embd}}}$ and projection layers for Keys and Values.
- [ ] Compute softmax weights over sequence lengths to yield pooled representations.
- [ ] Integrate this into `MultiTaskProteinClassifier` to replace standard mean pooling.

### Phase 2.5: Shared Latent Bottleneck Layer
- [ ] Implement a shared projection layer (e.g. `nn.Sequential` with `nn.Linear`, `nn.LayerNorm`, and `GELU`) in `MultiTaskProteinClassifier`.
- [ ] Connect the output of the `AttentionPooling` module to this shared latent layer.
- [ ] Branch the Pfam, EC, and stability classifier heads off the output of this shared bottleneck layer.

### Phase 3: Training & Multi-Task Convergence
- [ ] Train a smoke model using bidirectional attention, attention pooling, and shared latent layers on the multitask val dataset.
- [ ] Verify validation loss convergence on Pfam classification and EC function prediction.
- [ ] Verify that model checkpointing saves and loads attention pooling weights and shared projection parameters correctly.

### Phase 4: Saliency Mapping & Validation
- [ ] Write a script/utility (or update `scripts/eval_multi_task_critic.py`) to extract attention pooling weights $\alpha$ for a given sequence.
- [ ] Confirm that active sites or known motifs exhibit high attention scores.

---

## Done Definition

- Causal masking can be toggled off in the classifier config.
- `AttentionPooling` successfully replaces average pooling.
- Shared latent layer bottleneck is integrated and regularizes predictions.
- Saliency weights are extractable for downstream functional interpretation.
- Multi-task accuracy equals or exceeds the baseline causal model.

