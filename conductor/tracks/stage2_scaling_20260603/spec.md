# Specification: Stage 2 Data Scaling & Transfer Learning

## Overview
This track implements the first phase of "Stage 2 – Mid-Scale" from the project roadmap. It focuses on expanding the CodonLM training dataset with diverse bacterial genomes (Gram-positive and High-GC) and building reusable infrastructure for Transfer Learning.

## Objectives
1.  **Dataset Diversification:** Extract and incorporate genomes from `data/raw/gram_pos` and `data/raw/high-gc` into a unified training pack alongside existing Enterobacteriaceae data.
2.  **Transfer Learning Pipeline:** Modify the core training script to support fine-tuning from pre-trained weights without carrying over stale training state (optimizers, epochs).
3.  **Local Hardware Alignment:** Ensure the new training configurations respect the MacBook Pro M2/8GB RAM constraints.

## Technical Details
- **Data:** `src/codonlm/extract_cds_from_genbank.py` and `src/codonlm/build_dataset.py` will be used to process and pack the new unified dataset into `data/processed/stage2_diverse/`.
- **Training Script:** `src/codonlm/train_codon_lm.py` currently supports `--resume` (which restores full training state). We will add `--transfer_from <checkpoint>` which selectively loads only the model's `state_dict`, resetting the optimizer, scheduler, and epoch counter for a fresh fine-tuning run.
- **Configuration:** A new `configs/stage2_diverse.yaml` will be created to orchestrate this run.
