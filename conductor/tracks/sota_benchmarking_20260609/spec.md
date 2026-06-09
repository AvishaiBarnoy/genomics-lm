# SOTA Benchmarking Specification

## 1. Overview
This track defines the benchmarking framework to evaluate our models (CodonLM and ProteinLM/Critic) against State-of-the-Art (SOTA) genomic foundation models. Since our CodonLM is trained on prokaryotic genomes (*E. coli*, *Klebsiella*, *Salmonella*), the most appropriate SOTA comparison is **Evo 1** (which was also trained on prokaryotic and phage genomes). 

This document outlines the specific prokaryotic evaluation benchmarks used for Evo 1, SOTA metrics, and our local comparative evaluation strategy.

---

## 2. Baseline SOTA Models
We compare against four prominent SOTA genomic foundation models:

| Model | Architecture | Parameters | Pre-training Context | Primary Domain |
| :--- | :--- | :--- | :--- | :--- |
| **Evo 1** | StripedHyena (Hybrid) | 1.8B | 131,072 bp | **Prokaryotic & Phage** (OpenGenome: 2.7M genomes) |
| **DNABERT-2** | Transformer (ALiBi, BPE) | 117M | 512 - 10,000 bp | **Multi-Species** (Both Eukaryotic & Prokaryotic) |
| **HyenaDNA** | Hyena Operators (Sub-quadratic) | ~5M to 50M | 1,000 to 1,000,000 bp | **Eukaryotic-skewed** (Human Reference Genome) |
| **Caduceus** | Mamba (Bi-directional, RC) | 20M to 225M | 1,000 to 131,000 bp | **Eukaryotic-skewed** (Human Reference Genome) |

---

## 3. Pre-training Hardware & Compute Footprint
Comparing pre-training hardware reveals the resource density differences:

*   **Evo 1 (1.8B):** Pre-trained on **8 NVIDIA A100 GPUs** for 2–3 weeks.
*   **DNABERT-2 (117M):** Pre-trained on **8 NVIDIA A100 GPUs** for 2–3 days.
*   **Our Model (TinyGPT/ProteinLM):** Pre-trained on **1 Apple M2 Mac (8GB Unified RAM, MPS GPU)** for a few hours/days.

---

## 4. Target Benchmarks: Evo 1 Prokaryotic Suite
To evaluate our models against Evo 1 under scientific alignment, we use the following prokaryotic datasets:

### A. Zero-Shot Protein Mutational Fitness (Prokaryotic DMS)
*   **Description:** Predicting the fitness effects of single-point mutations on bacterial protein-coding genes without task-specific supervision.
*   **Datasets:** Bacterial Deep Mutational Scanning (DMS) datasets.
*   **Evaluation:** Zero-shot log-likelihood scoring of mutated sequences vs. wild-type sequences.

### B. Non-Coding RNA (ncRNA) Mutational Fitness
*   **Description:** Predicting mutational consequences on non-protein coding RNA.
*   **Dataset:** *E. coli* 5S ribosomal RNA (5S rRNA) DMS dataset.
*   **Evaluation:** Zero-shot sequence likelihoods compared to experimental fitness scores.

### C. Regulatory DNA & Gene Expression (Promoters / RBSs)
*   **Description:** Evaluating sequence composition effects on expression levels.
*   **Dataset:** **Kosuri et al.** promoter and ribosome binding site (RBS) library.
*   **Evaluation:** Correlation between model sequence likelihoods (or linear probes on top of embeddings) and experimental expression levels.

### D. Gene Essentiality
*   **Description:** Binary classification of whether a gene sequence is essential or non-essential for prokaryotic survival.
*   **Datasets:**
    1.  **Lambda Phage Essentiality** (Piya et al., 2023).
    2.  ***Pseudomonas aeruginosa* Essentiality** (Turner et al., 2015).
*   **Evaluation:** Sequence level classification using static embeddings + linear probes.

---

## 5. Evaluation Plan for our CodonLM & ProteinLM
1.  **Zero-Shot Likelihood Scoring:** Compute sequence perplexity/log-likelihoods for the wild-type and mutated variants of *E. coli* 5S rRNA and prokaryotic proteins. Compare rank correlation (Spearman's $\rho$) against Evo 1 results.
2.  **Downstream Probes:** Extract mean-pooled sequence embeddings from our backbones, train simple linear classifiers for lambda phage gene essentiality, and compare prediction F1 scores against SOTA.
3.  **Efficiency Density Ratio:** Evaluate accuracy / parameter count and accuracy / training-hours to show our M2 Mac optimized density performance.
