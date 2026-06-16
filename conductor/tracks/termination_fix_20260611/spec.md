# Termination Fix (2026-06-11)

## Track Metadata
- **ID:** `termination_fix_20260611`
- **Objective:** Fix the model's inability to place stop codons correctly by addressing data tokenization bugs, context window limitations, and signal dilution.
- **Status:** Planning

## Background & Motivation
An audit of the current data pipeline and model behavior revealed three major causes for the "0% termination rate" problem:
1.  **Tokenization Bug (Critical):** The pre-processed `data/processed/codon_ids.txt` file currently ends sequences with ID `3` (`<SEP>`) instead of ID `2` (`<EOS_CDS>`). The model is being trained to terminate genes with a separator rather than an explicit end-of-gene signal.
2.  **Context Window Truncation:** 13% of the sequences are longer than the 512-codon `block_size`. The current `pack_multi` and `pack_single` modes truncate these genes, hiding their biological termination entirely.
3.  **Signal Dilution:** The termination signal (one stop codon per sequence) is drowned out by the vast number of "continue" tokens.

## Scope & Impact
This track modifies the core CodonLM data preparation and training loop.
- **Tokenization:** Re-run the tokenizer to ensure perfect alignment with the vocabulary.
- **Data Loading:** Replace fixed-matrix NPZ packing with dynamic collate padding. This allows the model to process full-length genes without discarding data or wasting memory on fixed padding.
- **Loss Calculation:** Introduce a custom weighted CrossEntropyLoss to upweight terminal tokens, forcing the optimizer to prioritize correct termination.

## Proposed Solution
1.  **Fix Tokenization:** Execute `src/codonlm/codon_tokenize.py` with `--termination eos` to create `data/processed/codon_ids_v2.txt` from the 9-genome Master set (`stage2_dna.txt`), ensuring `<EOS_CDS>` is properly placed.
2.  **Dynamic Collate:**
    - Update `src/codonlm/build_dataset.py` to save raw lists of tokenized sequences instead of pre-padded `X` and `Y` matrices.
    - Implement a custom `collate_fn` in `train_codon_lm.py` that dynamically pads sequences to the maximum length within each specific batch.
3.  **Loss Upweighting:**
    - Update `model_tiny_gpt.py` or the training loop in `train_codon_lm.py` to apply a weight multiplier (e.g., 5.0x) to the indices corresponding to `<EOS_CDS>`, `TAA`, `TAG`, and `TGA` during loss computation.