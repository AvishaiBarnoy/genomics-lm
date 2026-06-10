# Specification: Stage 2.6 – Multi-Scale Biophysical Architecture

This track specifies the integration of local physical nucleotide constraints with global codon syntax and amino acid semantics, resolving the structural limitations discovered during the termination motif audit of the Stage 2.5 model.

---

## 1. Background & Rationale
Our diagnostics (`docs/termination_motifs_analysis.md`) showed that CodonLM lacks native generation of secondary mRNA structure motifs (hairpins, poly-T tracts) downstream of stop codons. This is due to:
* **Codon Smoothing**: Triplets smooth over individual nucleotide base-pairing symmetries.
* **Directional Causal Bias**: Autoregressive left-to-right generation prevents UTR terminators from guiding the upstream stop-codon placement.
* **CDS-Centric Bias**: The training set is heavily dominated by coding sequences rather than regulatory non-coding regions.

Moving to a full nucleotide model increases token count by $3\times$ and attention computation by $9\times$ ($O(L^2)$). This track designs a multi-scale biophysical architecture to capture these structures under local hardware constraints (8GB RAM).

---

## 2. Structural Requirements

### A. Dual-Track Late Fusion (Structural Compass)
* **Goal**: Embed structural directionality into the codon-level generator using a sliding-window nucleotide encoder.
* **Nucleotide Encoder (Layer 1)**: A lightweight CNN or local Transformer scanning local windows of 60 bp. It outputs continuous shape/energy embeddings representing physical properties (MGW, Roll, EP).
* **Codon Generator (Layer 2)**: Learns to condition its autoregressive predictions on the physical shape vectors supplied by the encoder (via cross-attention or embedding injection).
* **Overhead Target**: $<15\%$ inference computational overhead.

### B. Hybrid Tokenization (Variable-Scale Tape)
* **Goal**: Enable single-nucleotide resolution *only* at regulatory boundaries (promoters, operators, terminators) while keeping coding regions compressed.
* **Vocabulary**: Blend of 64 codons + 4 single nucleotides + structural boundary tags.
* **Grammar**: The dataset tokenizes coding sequences (CDS) as codons ($3\times$ compression) and intergenic/UTR regions as single nucleotides (1 bp resolution).

### C. Energy-Based mRNA Optimizer (EBM)
* **Goal**: Train a global, bidirectional model to optimize synonymous codon sequences for thermodynamic stability.
* **Boltzmann Distribution**: $P(x) \propto e^{-E(x)/kT}$ where energy directly maps to folding free energy ($\Delta G$).
* **Langevin Dynamics**: Use gradient-guided MCMC to optimize candidate sequences by lowering structural energy.
