# Implementation Plan: Large Data-Scaling for Taxonomic Diversity

## Phase 1: Data Selection and Acquisition
Download diverse bacterial genomes to build the expanded corpus.

- [x] Task: Create accession manifest `data/accession_list.txt`.
    - Select 30–50 representative bacterial genomes covering 15–20 distinct families.
    - Balance GC content (30% to 75%) and genome sizes.
- [x] Task: Implement `scripts/download_genomes.py`.
    - Automate downloading `.gbff` files from NCBI using the accessions.
    - Verify file integrity and organize them under `data/raw/`.

## Phase 2: Expanded Corpus Packaging
Extract and package the master dynamic datasets.

- [x] Task: Run GenBank extraction.
    - Execute `extract_cds_from_genbank.py` for all downloaded genomes.
    - Consolidate all extracted coding sequences into a single `stage2.6_large_master_dna.txt` and metadata file `stage2.6_large_master_meta.tsv`.
- [x] Task: Tokenize and Build NPZ splits.
    - Run `codon_tokenize.py` with `--termination eos` to tokenize the master DNA corpus.
    - Run `build_dataset.py` with `--pack_mode dynamic --block_size 512` to create the train/val/test splits.

## Phase 3: Scaling Model Training
Train and evaluate the scaled model.

- [x] Task: Configure training parameters.
    - Create `configs/stage2.6_large_scaling.yaml` pointing to the new NPZ splits.
    - Configure batch size 4 and grad accumulation 32 to fit M2 memory limits.
- [x] Task: Causal Attention SDPA Memory Optimization for MPS.
    - Optimize [CausalSelfAttention](file:///Users/User/github/genomics-lm/src/codonlm/model_tiny_gpt.py#L9) to pass `attn_mask=None` and `is_causal=True` when `attn_mask` is not provided to trigger the highly optimized fused causal kernel.
    - Precompute combined causal and segment masks in [TinyGPT.forward](file:///Users/User/github/genomics-lm/src/codonlm/model_tiny_gpt.py#L114) and pass down to reduce block-level duplicate allocations.
- [ ] Task: Train the model.
    - Run `train_codon_lm.py` initializing weights from the best `6L4H_d384_transfer` checkpoint.
    - Train for 5–10 epochs and run full evaluations using `evaluate_run.py`.
