# SOTA Benchmarking Specification

## 1. Overview
This track defines the benchmarking framework to evaluate our models (CodonLM and ProteinLM/Critic) exclusively against State-of-the-Art (SOTA) **prokaryotic foundation models** (such as Evo 1 and GenSLM). Since our models are trained and designed for prokaryotic genomic sequence tasks, comparing against eukaryotic-focused models is omitted to ensure scientific alignment.

---

## 2. Baseline SOTA Prokaryotic Models
We compare our models against established prokaryotic foundation models:

| Model | Architecture | Parameters | Pre-training Dataset | Primary Domain |
| :--- | :--- | :--- | :--- | :--- |
| **Evo 1** | StripedHyena (Hybrid) | 1.8B | **OpenGenome** (2.7M prokaryotic & phage genomes, 300B nucleotides) | **Prokaryotic & Phage** (Genomic level) |
| **GenSLM** | GPT-style Transformer | 25M to 2.5B | **BV-BRC** (110M prokaryotic/viral gene sequences) | **Prokaryotic & Viral** (Genomic/Gene level) |
| **ProGen2** | Transformer Decoder | 110M to 6.4B | **UniRef / BFD** (280M protein sequences) | **Multi-Domain** (Protein level, prokaryotic-rich) |

---

## 3. Pre-training Hardware & Compute Footprint
To evaluate resource-efficiency density, we compare pre-training hardware profiles:

*   **Evo 1 (1.8B):** Pre-trained on **8 NVIDIA A100 (80GB) GPUs** for 2–3 weeks.
*   **GenSLM (2.5B):** Pre-trained on **up to 512 NVIDIA A100 GPUs** (on Polaris/ThetaGPU supercomputers).
*   **Our Model (TinyGPT/ProteinLM):** Pre-trained on **1 Apple M2 Mac (8GB Unified RAM, MPS GPU)** for a few hours/days.

---

## 4. Target Benchmarks: Prokaryotic Evaluation Suite
Our models are evaluated against the specific prokaryotic benchmarks established in SOTA prokaryotic model literature:

### A. Zero-Shot Protein Mutational Fitness (Prokaryotic DMS)
*   **Description:** Predicting the fitness consequences of single-point mutations on bacterial protein-coding genes without task-specific supervision.
*   **Datasets:** Prokaryotic Deep Mutational Scanning (DMS) datasets.
*   **Evaluation:** Zero-shot log-likelihood scoring of mutated vs. wild-type sequences.

### B. Non-Coding RNA (ncRNA) Mutational Fitness
*   **Description:** Predicting functional consequences of mutations on non-protein coding bacterial RNA.
*   **Dataset:** *E. coli* 5S ribosomal RNA (5S rRNA) DMS dataset.
*   **Evaluation:** Rank correlation between zero-shot sequence log-likelihoods and experimental fitness scores.

### C. Regulatory DNA & Gene Expression (Promoters / RBSs)
*   **Description:** Evaluating sequence composition effects on expression levels.
*   **Dataset:** **Kosuri et al.** promoter and ribosome binding site (RBS) library.
*   **Evaluation:** Correlation between model sequence likelihoods (or linear probes on top of embeddings) and experimental expression levels.

### D. Gene Essentiality
*   **Description:** Binary classification of whether a gene sequence is essential or non-essential for prokaryotic survival.
*   **Datasets:**
    1.  **Lambda Phage Essentiality** (Piya et al., 2023).
    2.  ***Pseudomonas aeruginosa* Essentiality** (Turner et al., 2015).
*   **Evaluation:** Sequence-level classification using static embeddings + linear probes.

---

## 5. Evaluation Plan for our CodonLM & ProteinLM
1.  **Zero-Shot Likelihood Scoring:** Compute sequence perplexity/log-likelihoods for the wild-type and mutated variants of *E. coli* 5S rRNA and prokaryotic proteins. Compare rank correlation (Spearman's $\rho$) against Evo 1 and GenSLM results.
2.  **Downstream Probes:** Extract mean-pooled sequence embeddings from our backbones, train simple linear classifiers for lambda phage gene essentiality, and compare prediction F1 scores against SOTA.
3.  **Compute Efficiency Density Ratio:** Evaluate accuracy / parameter count and accuracy / training-hours to show our M2 Mac optimized density performance.
