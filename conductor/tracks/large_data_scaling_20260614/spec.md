# Specification: Large Data-Scaling for Taxonomic Diversity

## Overview
Our current master dataset (`stage2.5_master`) contains genomes representing 3 broad categories (Enterobacteriaceae, Gram-positives, and High-GC Actinobacteria) but consists of only 9 unique genomes total. This narrow database limits the model's vocabulary and leads to overfitting on family-specific codon usage dialects.

This track scales the pre-training dataset to **30–50 unique genomes** representing **15–20 distinct taxonomic families**. Scaling data diversity is crucial to force the model to decouple amino acid semantics from specific codon usage patterns, enabling it to learn a generalized biophysical representation of genomics across a broad GC-content spectrum.

## Objectives
1. **Diverse Data Acquisition**: Download GenBank (.gbff) files for 30–50 representative genomes across 15–20 distinct bacterial families (e.g. adding *Pseudomonadaceae*, *Lactobacillaceae*, *Clostridiaceae*, *Spirochaetaceae*, etc.).
2. **Master Corpus Generation**: Build a large consolidated master corpus containing ~150k–200k genes (~50M–75M tokens) and GC contents ranging from 30% to 75%.
3. **Dynamic Dataset Packing**: Pack the dataset in `dynamic` mode with natural-length gene windows at a block size of 512, preserving natural stop codons and `<EOS_CDS>` markers.
4. **Memory Optimization**: Implement true fused causal attention (SDPA) when segment masks are disabled, and precompute combined masks to avoid layer-level redundant allocations on MPS.
5. **Scale Pre-training**: Train the `10L8H_d384` transfer model on the scaled dataset for 5–10 epochs, establishing a high-diversity baseline within the M2 Mac memory limits.

## Technical Details
* **Acquisition Tool**: `scripts/download_genomes.py` using NCBI datasets CLI or raw FTP access.
* **Corpus Pipeline**: Run extraction (`extract_cds_from_genbank.py`), tokenization (`codon_tokenize.py`), and dataset packaging (`build_dataset.py` with `--pack_mode dynamic`) to output `stage2.6_large_master` train/val/test splits.
* **Hardware Constraints**: Ensure batch size and gradient accumulation maintain an effective batch size of 128 (e.g. batch size 4, grad accum 32) to prevent Apple Silicon MPS OOM paging. Enable `use_sdpa: true` with true fused causal execution paths to minimize peak activation storage on MPS.
