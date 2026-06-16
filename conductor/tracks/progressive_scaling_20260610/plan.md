# Plan: Stage 2.7 – Progressive High-Capacity Scaling Ladder

This plan details the implementation milestones to execute progressive transfer learning for high-capacity CodonLM configurations.

---

## Status Update (2026-06-16)
We have successfully implemented and executed the model scaling ladder for the **`d_embd: 384`** series:
- **`4L2H_d384`** (Base model trained from scratch)
- **`6L4H_d384`** (Layer-transfer from `4L2H`, trained successfully)
- **`10L8H_d384`** (Layer-transfer from `6L4H`, trained successfully)
- **Stage 2.6 `10L8H_d384` on the expanded 15-genome corpus** became the current best CodonLM (`2026-06-15_stage2.6_10L8H_d384_e10`).

Because `d_embd` was kept constant at 384, the linear projections in attention heads had identical tensor shapes `(384, 384)`. This bypassed the need for tensor resizing, allowing PyTorch's native `load_state_dict(..., strict=False)` to transfer weights seamlessly (leaving only the new layers initialized randomly).

The advanced weight reshaper script in Phase 2 is still required for scaling across different embedding dimensions (e.g. `d_embd: 384` to `d_embd: 512`).

### Current d384 Evidence

| Run | Role | Key result |
|---|---|---|
| `2026-06-12_stage2.5_4L2H_d384_e5` | base d384 model | trained from scratch |
| `2026-06-12_stage2.5_6L4H_d384_e5` | layer-expanded transfer | last ppl 84.04; regression probe avg_r2 0.552 |
| `2026-06-12_stage2.5_10L8H_d384_e5` | high-capacity transfer | trained; regression probe avg_r2 0.541 on stage2.5 data |
| `2026-06-15_stage2.6_10L8H_d384_e10` | expanded-corpus best model | test ppl 68.53; regression probe avg_r2 0.569; EC/AMR conference probes completed |

### d512 Decision

Do **not** promote d512 to the mainline yet. A `10L8H_d512` model is about 29.2M params versus about 16.5M for `10L8H_d384` (~1.77x), and cross-width transfer still needs `scripts/expand_model.py`. If we test d512, start with a controlled pilot (`6L8H_d512` or `8L8H_d512`) and compare validation loss, structural probes, and generation metrics before committing to a full 10-layer run.

---

## Task List & Milestones

### Phase 1: Base Model Training (Milestone 1)
- [x] **Task 1.1: Configure Stage 1 (`4L2H_d384` / `4L2H_d512`)**
  - Completed for `d_embd: 384` (`configs/stage2.5_d384_dynamic.yaml`).
- [x] **Task 1.2: Train Stage 1 Base**
  - Trained `2026-06-12_stage2.5_4L2H_d384_e5` successfully.
- [x] **Task 1.3: Run Diagnostic Pipeline**
  - Run regression probes, attention analyses, and dialect assessments on the base model.

### Phase 2: Layer & Head Expansion (Milestone 2)
- [ ] **Task 2.1: Implement Weight Reshaper & Transporter**
  - Write a utility script `scripts/expand_model.py` to map checkpoint weights across different embedding dimensions (e.g., query/key/value projection mapping and FFN expansion).
- [ ] **Task 2.2: Verify Shape Alignment**
  - Write unit tests in `tests/test_model_expansion.py` to assert that the expanded model loads successfully and outputs identical predictions prior to fine-tuning.

### Phase 3: High-Capacity Fine-Tuning (Milestone 3)
- [x] **Task 3.1: Train Stage 2 (`6L4H_d384` / `6L4H_d512`)**
  - Completed for `d_embd: 384` (`stage2.5_6L4H_d384_transfer`).
- [x] **Task 3.2: Train Stage 3 (`10L8H_d384` / `10L8H_d512`)**
  - Completed for `d_embd: 384` (`2026-06-12_stage2.5_10L8H_d384_e5`), then continued on the expanded Stage 2.6 corpus (`2026-06-14` / `2026-06-15` runs).
- [ ] **Task 3.3: Compare Scaling Performance**
  - Generate a comparative summary of performance metrics across all d256/d384 stages using `compare_runs.py`.
  - Add a small d512 pilot only after the cross-width expansion utility exists.
