# Implementation Plan: Genomic Tape Extraction

## Phase 1: Tape Extraction Logic [checkpoint: ]
Implement the core logic to extract contiguous genomic segments.

- [x] Task: Create `src/codonlm/extract_genomic_tape.py`.
    - [x] Implement sliding window extraction across the entire chromosome.
    - [x] Support handling of both forward and reverse strands (extracting the sense strand relative to the reference).
    - [x] Filter segments that contain non-ACGT characters (e.g., Ns).
- [x] Task: Coordinate Tracking.
    - [x] Implement a metadata output that tracks (genome_id, start_bp, end_bp, strand).

## Phase 2: Tokenization & Training Integration [checkpoint: ]
Prepare the tape data for the model.

- [x] Task: Tokenize Tapes.
    - [x] Use `src/codonlm/codon_tokenize.py` on the tape data.
- [x] Task: Pack Tape Dataset.
    - [x] Use `src/codonlm/build_dataset.py` with `pack_mode='single'` to maintain contiguous chromosomal context within each model window.

## Phase 3: Validation & Training [checkpoint: ]
- [x] Task: Verification.
    - [x] Verify that tape segments correctly overlap known operons and transitions.
- [/] Task: Launch Stage 2.5 "Master" Training. (In Progress)
    - [x] Execute fine-tuning from pre-trained 6-layer weights on the mixed Tape+Bridge dataset.
