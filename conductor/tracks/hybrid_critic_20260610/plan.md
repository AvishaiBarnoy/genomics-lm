# Hybrid DNA-Protein Critic Benchmark Implementation Plan

## Phase 1: Guided Autoregressive Generation
- **Task 1.1: Partial Sequence Evaluation**
  - Implement dynamic codon-to-amino-acid translation during autoregressive generation for incomplete sequences.
- **Task 1.2: Logit Blending Logic**
  - Implement a configurable weight parameter ($\alpha$) to blend next-token logits from CodonLM and partial-sequence logits from the Multi-Task Critic.

## Phase 2: Evaluation & Benchmarking
- **Task 2.1: Benchmarking Script**
  - Create `scripts/benchmark_hybrid_critic.py` to compare generation throughput, early-termination rates, stability yields, and structural diversity across sampling styles.
- **Task 2.2: SOTA Alignment**
  - Profile the hybrid model against published prokaryotic baselines and document the compute efficiency density.

## Phase 3: Dashboard Integration & Tests
- **Task 3.1: Streamlit Dashboard Tab**
  - Integrate hybrid guidance toggles into the Model Playground tab.
- **Task 3.2: Verification Tests**
  - Add unit tests verifying logit blending bounds and partial translation correctness.
