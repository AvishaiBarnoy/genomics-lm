# SOTA Benchmarking Implementation Plan

This plan outlines the steps required to execute baseline benchmarking evaluations on our local models and log comparative results.

---

## Phase 1: Research & Setup
- [ ] **Task 1.1:** Fetch exact downstream performance tables from SOTA papers:
  - DNABERT-2 (Zhou et al., 2024) - GUE benchmark tables.
  - HyenaDNA (Nguyen et al., 2023) - GenomicBenchmarks tables.
  - Caduceus (Schiff et al., 2024) - Variant effect and epigenetic classification tables.
- [ ] **Task 1.2:** Install `GenomicBenchmarks` package:
  ```bash
  pip install genomic-benchmarks
  ```
- [ ] **Task 1.3:** Setup downloading and preprocessing scripts for GUE sample datasets (e.g., promoter and splice-site datasets).

---

## Phase 2: Feature Extraction Pipeline
- [ ] **Task 2.1:** Implement a unified feature extractor script `scripts/benchmark_extract_features.py` that:
  - Loads a trained backbone (CodonLM TinyGPT or ProteinLM backbone).
  - Encodes sequences from the benchmark dataset.
  - Generates token-level embeddings and applies mean pooling to output sequence-level feature arrays (`.npz` files).
- [ ] **Task 2.2:** Support sliding-window feature pooling for long context inputs exceeding default context lengths.

---

## Phase 3: Linear Probe Evaluation
- [ ] **Task 3.1:** Write a evaluation runner script `scripts/benchmark_eval_heads.py` to:
  - Load the generated feature arrays (`.npz` files).
  - Train a simple PyTorch linear probe or 1-layer MLP classification head.
  - Calculate standard metrics (Accuracy, MCC, F1 Score).
- [ ] **Task 3.2:** Save metrics output to `runs/<run_id>/scores/sota_benchmarks.json`.

---

## Phase 4: Comparative Reporting
- [ ] **Task 4.1:** Write a reporting script `scripts/generate_sota_report.py` to:
  - Read local metrics from `runs/<run_id>/scores/sota_benchmarks.json`.
  - Fetch stored literature benchmarks for DNABERT-2, HyenaDNA, and Caduceus.
  - Calculate and report the *compute efficiency ratio* (Performance score divided by training hardware energy/FLOPs).
  - Generate a markdown comparison table in the dashboards.
