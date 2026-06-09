# SOTA Benchmarking Specification

## 1. Overview
This track defines the benchmarking framework to evaluate our models (CodonLM and ProteinLM/Critic) against State-of-the-Art (SOTA) genomic foundation models. The goals are to identify standard biological benchmarking suites, log the performance of SOTA architectures, compare our resource efficiency (pre-training hardware & parameters), and outline downstream evaluation tasks.

---

## 2. Baseline SOTA Models
We compare against four prominent SOTA genomic foundation models:

| Model | Architecture | Parameters | Pre-training Context | Primary Strengths |
| :--- | :--- | :--- | :--- | :--- |
| **DNABERT-2** | Transformer (ALiBi, BPE) | 117M | 512 - 10,000 bp | Highly efficient baseline, strong on human genome tasks. |
| **HyenaDNA** | Hyena Operators (Sub-quadratic) | ~5M to 50M | 1,000 to 1,000,000 bp | Single-nucleotide resolution at massive context scales. |
| **Caduceus** | Mamba (Bi-directional, RC-Equivariant) | 20M to 225M | 1,000 to 131,000 bp | Handles reverse-complement equivariance natively. |
| **Evo (Evo 2)** | StripedHyena / Hybrid | Up to 40B | 8,192 to 131,072 bp | Massive context generative capabilities, genome-scale modeling. |

---

## 3. Pre-training Hardware & Compute Footprint
To contextualize our model performance, we track the massive scale difference in pre-training hardware:

*   **Evo 2 (40B):** Pre-trained on **>2,000 NVIDIA H100 GPUs** for several months.
*   **Nucleotide Transformer (2.5B):** Pre-trained on **128 NVIDIA A100 GPUs** for 28 days.
*   **Caduceus (225M):** Pre-trained on **8 NVIDIA H100 GPUs** for 25 days.
*   **HyenaDNA (Large):** Pre-trained on a **single node of 8 NVIDIA A100 GPUs** for 2–3 weeks.
*   **DNABERT-2 (117M):** Pre-trained on **8 NVIDIA A100 GPUs** for 2–3 days.
*   **Our Model (TinyGPT/ProteinLM):** Pre-trained on **1 Apple M2 Mac (8GB Unified RAM, MPS GPU)** for a few hours/days.

---

## 4. Standard Biological Benchmarks
We evaluate models across two primary genomic benchmarking suites:

### A. GUE (Genomic Understanding Evaluation)
Introduced with DNABERT-2, GUE evaluates models across 36 datasets:
1.  **Variant Effect Prediction (VEP):** Predicting function consequences of single-nucleotide changes.
2.  **Promoter Detection:** Identifying transcription start sites (mammalian/core promoters).
3.  **Enhancer & Promoter Interactions:** Mapping distant regulatory interactions.
4.  **Splice Site Prediction:** Identifying donor and acceptor sites.
5.  **Epigenetic Modifications:** Histone mark and chromatin accessibility classification.

### B. GenomicBenchmarks
A lightweight package consisting of classification tasks:
1.  **Coding vs. Non-Coding:** Distinguishing exons/introns from intergenic regions.
2.  **Enhancer Detection:** Distinguishing active enhancer regions from random genomic sequences.
3.  **Species Classification:** Differentiating sequence pieces across diverse biological domains.

---

## 5. Evaluation Strategy for Our Models
To evaluate our local models on these benchmarks:
*   **Feature Extraction:** Run our trained model backbones on the benchmark inputs to generate embeddings (utilizing mean pooling over tokens).
*   **Downstream Probes:** Train simple Linear/MLP classification heads on top of these static embeddings to compute downstream classification scores (Accuracy, MCC, or F1).
*   **Resource efficiency ratio:** Measure accuracy per parameter and per compute-hour to show performance density.
