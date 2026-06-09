# SOTA Benchmarking Implementation Plan

This plan outlines the steps required to evaluate our local prokaryotic models against Evo 1 using standard prokaryotic evaluation suites and stress-test our local hardware.

---

## Phase 0: Local Pre-training Dataset Scaling (Hardware Stress-Test)
- [ ] **Task 0.1:** Curate a subset of 100–200 representative prokaryotic genomes from GenBank (targeting 300M–500M nucleotides).
- [ ] **Task 0.2:** Write a pipeline preparation script to extract and tokenize all CDS coding sequences.
- [ ] **Task 0.3:** Conduct a continuous 24–48 hour training run (stress-test) on the local M2 Mac GPU to establish a scratch-pretrained baseline.

---

## Phase 1: Benchmark Data Acquisition
- [ ] **Task 1.1:** Retrieve Deep Mutational Scanning (DMS) datasets for prokaryotic proteins and *E. coli* 5S rRNA (referenced in Stanford Arc's Evo 1 papers).
- [ ] **Task 1.2:** Download the Kosuri promoter/RBS expression datasets.
- [ ] **Task 1.3:** Retrieve gene essentiality labels for:
  - Lambda phage essentiality (Piya et al., 2023).
  - *Pseudomonas aeruginosa* essentiality (Turner et al., 2015).

---

## Phase 2: Zero-Shot Mutation Scoring Pipeline
- [ ] **Task 2.1:** Implement a zero-shot scoring script `scripts/benchmark_zero_shot_mutations.py` that:
  - Loads our CodonLM model.
  - Takes a wild-type and a mutated nucleotide sequence.
  - Computes the log-likelihood (or perplexity difference) under the model.
  - Calculates the Spearman rank correlation against experimental DMS fitness values.
- [ ] **Task 2.2:** Verify that the scoring handles codon boundaries and sequence padding correctly.

---

## Phase 3: Gene Essentiality Classification
- [ ] **Task 3.1:** Implement `scripts/benchmark_gene_essentiality.py` to:
  - Extract sequence embeddings from our backbones for the Lambda phage and *P. aeruginosa* gene datasets.
  - Train a simple linear probe on the embeddings.
  - Report classification metrics (Accuracy, F1, MCC).

---

## Phase 4: Comparative Reports
- [ ] **Task 4.1:** Consolidate local metrics and compare them to published Evo 1 results (DMS correlation, essentiality F1).
- [ ] **Task 4.2:** Calculate performance efficiency density (e.g. F1 score divided by parameter size and GPU pre-training hours) to compare M2 Mac efficiency vs. A100 pre-training.
