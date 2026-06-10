# SOTA Benchmarking Implementation Plan

This plan outlines the steps required to evaluate our local prokaryotic models against Evo 1 and GenSLM using standard prokaryotic evaluation suites.

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
- [ ] **Task 4.1:** Consolidate local metrics and compare them to published Evo 1 and GenSLM results (DMS correlation, essentiality F1).
- [ ] **Task 4.2:** Calculate performance efficiency density (e.g. F1 score divided by parameter size and GPU pre-training hours) to compare M2 Mac efficiency vs. A100/H100 pre-training.

---

## Phase 5: Future Hybrid DNA-Protein Critic Evaluation
- [ ] **Task 5.1:** Integrate the Multi-Task Protein Critic as a bidirectional re-feeding evaluator.
- [ ] **Task 5.2:** Score non-synonymous mutations using Critic stability logits combined with CodonLM synonymous likelihoods.
- [ ] **Task 5.3:** Concatenate DNA and protein embeddings to train downstream hybrid essentiality classifiers.

