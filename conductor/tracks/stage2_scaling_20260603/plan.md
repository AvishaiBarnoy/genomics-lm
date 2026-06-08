# Implementation Plan: Stage 2 Data Scaling & Transfer Learning

## Phase 1: Transfer Learning Infrastructure [checkpoint: ]
Modify the training architecture to support clean transfer learning.

- [x] Task: Update `src/codonlm/train_codon_lm.py`.
    - [x] Add `--transfer_from` command-line argument.
    - [x] Implement conditional loading logic: if `--transfer_from` is provided, load the checkpoint but ONLY restore `model.load_state_dict()`. Reset `start_epoch`, `step`, and do not load optimizer/scheduler states.
- [x] Task: Create `configs/stage2_diverse.yaml`.
    - [x] Define the configuration targeting the new `data/processed/stage2_diverse/` NPZ files.
    - [x] Maintain the `4L2H_d128` architecture to match the pre-trained weights.
- [x] Task: Update `ROADMAP.md` to indicate progress on Stage 2.

## Phase 2: Data Extraction & Packing [checkpoint: ]
Process the raw diverse genomes and build the unified dataset.

- [x] Task: Extract Gram-positive and High-GC genomes.
    - [x] Run `extract_cds_from_genbank.py` on `data/raw/gram_pos/*.gbff`.
    - [x] Run `extract_cds_from_genbank.py` on `data/raw/high-gc/*.gbff`.
- [x] Task: Combine and Pack the Dataset.
    - [x] Concatenate the new and existing `cds_dna.txt` files.
    - [x] Run `codon_tokenize.py` to generate token IDs.
    - [x] Run `build_dataset.py` to generate the randomized, cross-genome `train/val/test` NPZ packs.
- [x] Task: Pre-flight Audit & Verification.
    - [x] Run `scripts/audit_stage2_data.py` to verify taxonomic diversity.
    - [x] Run a 10-step "smoke test" training run to verify weight loading.

## Phase 3: Fine-Tuning Execution [checkpoint: ]
Launch the Stage 2 training run.

- [x] Task: Run the transfer learning process.
    - [x] Execute `train_codon_lm.py` with corrected architecture and diversified data.
- [x] Task: Run Long-Context (512-block) Training.
    - [x] Train a 6-layer model with doubled context size to improve gene termination.
