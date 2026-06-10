# Plan: Stage 2.7 – Progressive High-Capacity Scaling Ladder

This plan details the implementation milestones to execute progressive transfer learning for high-capacity CodonLM configurations.

---

## Task List & Milestones

### Phase 1: Base Model Training (Milestone 1)
- [ ] **Task 1.1: Configure Stage 1 (`4L2H_d512`)**
  - Create configuration file `configs/stage2.7_base_4L2H.yaml` with `d_embd: 512`, `n_layer: 4`, `n_head: 2`.
- [ ] **Task 1.2: Train Stage 1 Base**
  - Train the base model from scratch on the pre-processed dataset for 5-10 epochs.
- [ ] **Task 1.3: Run Diagnostic Pipeline**
  - Run the 6-step interpretability pipeline on Stage 1 to verify base AA identity and start/stop codon representations.

### Phase 2: Layer & Head Expansion (Milestone 2)
- [ ] **Task 2.1: Implement Weight Reshaper & Transporter**
  - Write a utility script `scripts/expand_model.py` that loads a checkpoint, duplicates attention heads (query, key, value projection weights), copies intermediate feed-forward and layer norm weights, and pads new layer blocks with identity mappings.
- [ ] **Task 2.2: Verify Shape Alignment**
  - Write unit tests in `tests/test_model_expansion.py` to assert that the expanded model loads successfully and outputs identical predictions to the base model on a sample batch prior to training.

### Phase 3: High-Capacity Fine-Tuning (Milestone 3)
- [ ] **Task 3.1: Train Stage 2 (`6L4H_d512`)**
  - Transfer weights from `4L2H` and train for 3-5 epochs.
- [ ] **Task 3.2: Train Stage 3 (`10L8H_d512`)**
  - Transfer weights from `6L4H` and train for 3-5 epochs.
- [ ] **Task 3.3: Compare Scaling Performance**
  - Generate a comparative summary of the performance metrics across all three stages using `compare_runs.py`.
