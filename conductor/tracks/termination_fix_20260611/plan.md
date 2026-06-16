# Implementation Plan: Termination Fix

## Objective
Implement dynamic collation, loss upweighting, and tokenization fixes to solve the gene termination problem.

## Key Files & Context
-   `src/codonlm/codon_tokenize.py`
-   `src/codonlm/build_dataset.py`
-   `src/codonlm/train_codon_lm.py`
-   `src/codonlm/model_tiny_gpt.py`

## Implementation Steps

### Phase 1: Data Re-Tokenization & Verification
1.  Run `python -m src.codonlm.codon_tokenize --inp data/processed/stage2_dna.txt --out_ids data/processed/codon_ids_v2.txt --termination eos` to regenerate the Master set with proper termination.
2.  Verify via shell script that sequences in `codon_ids_v2.txt` end with ID `2` (`<EOS_CDS>`).

### Phase 2: Dynamic Dataset Refactor
1.  **Modify `build_dataset.py`:**
    -   Add a `pack_dynamic` mode (or modify `pack_single`) that saves a serialized list of raw integer arrays (e.g., using `pickle` or `np.savez` with an object array) rather than fixed `N x block_size` matrices.
2.  **Modify `train_codon_lm.py`:**
    -   Update `PackedDataset` to load the new dynamic structure.
    -   Create a `dynamic_collate_fn(batch)` that takes a list of sequences, determines the max length `L` in the batch, and pads all sequences to `L` using `<PAD>` (ID 0).
    -   Ensure `X` and `Y` are correctly sliced (e.g., `x = seq[:-1]`, `y = seq[1:]`).
    -   Update the `DataLoader` initialization to use `collate_fn=dynamic_collate_fn`.

### Phase 3: Loss Upweighting
1.  **Modify `model_tiny_gpt.py` (or `train_codon_lm.py`):**
    -   Identify the indices for `TAA`, `TAG`, `TGA`, and `<EOS_CDS>`.
    -   Create a `weight` tensor of size `vocab_size` initialized to 1.0.
    -   Set the weight of the terminal indices to `5.0`.
    -   Pass this `weight` tensor to `F.cross_entropy(..., weight=weight)`.

## Verification & Testing
1.  Run a small debug training loop (`scripts/profile_train.py` or similar) to ensure the dynamic dataloader doesn't cause Out-Of-Memory errors on the M2.
2.  Verify the loss decreases properly.
3.  Perform inference (`generate.py`) on the newly trained checkpoint to measure the termination rate. It should rise from 0% to a non-zero, biologically viable percentage.