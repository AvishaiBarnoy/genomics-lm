# Plan: Stage 2.6 – Multi-Scale Biophysical Architecture

This plan details the implementation steps to execute the Stage 2.6 specification.

---

## Task List & Milestones

### Phase 1: Hybrid Tokenizer Prototype (Milestone 1)
- [x] **Task 1.1: Build Hybrid Tokenizer**
  - Implement a tokenizer class that accepts a genomic sequence with annotated coding (CDS) and intergenic (UTR) regions.
  - Tokenize CDS spans as 3-nucleotide codons, and UTR/intergenic spans as individual nucleotides (or BPE nucleotide blocks).
- [x] **Task 1.2: Build Training Pipeline Parser**
  - Adapt `src/codonlm/extract_cds_from_genbank.py` or pipeline scripts to keep 30 bp upstream of START and 60 bp downstream of STOP.
  - Generate a hybrid-tokenized dataset for training.
- [x] **Task 1.3: Add Unit Tests**
  - Verify boundary alignment, vocabulary sizes (68 tokens + 4 nucleotides = 72 tokens), and that the decoding reconstruction is lossless.

### Phase 1b: Physical Termination Transfer Pilot (Milestone 1b)
- [x] **Task 1b.1: Reuse existing hybrid-token pipeline**
  - Use `src.codonlm.extract_hybrid_from_genbank`, `src.codonlm.hybrid_tokenize`, and `src.codonlm.pipeline_prepare_hybrid`.
  - Keep CDS spans codon-tokenized and 3' downstream regions nucleotide-tokenized.
- [x] **Task 1b.2: Add transfer-learning config**
  - Config: `configs/physical_termination_transfer.yaml`.
  - Starting checkpoint: `runs/2026-06-18_termination_aux_mps_b4_v1/checkpoints/best.pt`.
  - Fine-tuning target: hybrid CDS + 30 bp upstream + 120 bp downstream windows.
  - Dataset scope: all local non-duplicate GBFF files from Enterobacteriaceae,
    Gram-positive, high-GC, and `data/raw/expanded`.
  - Hardware profile: MPS, `batch_size=4`, low LR, periodic checkpoints.
- [x] **Task 1b.3: Add tokenizer-expansion transfer support**
  - Codon-only checkpoints have fewer vocabulary rows than the hybrid tokenizer.
  - The trainer now copies same-shaped weights exactly and copies token embedding
    / output rows by token name where vocabularies overlap.
  - New UTR/nucleotide token rows remain initialized by the target model.
- [x] **Task 1b.4: Prepare the hybrid boundary dataset**
  - Command:
    `python -m src.codonlm.pipeline_prepare_hybrid --config configs/physical_termination_transfer.yaml --run-id 2026-06-19_physical_termination_transfer_mps_b4 --run-dir runs/2026-06-19_physical_termination_transfer_mps_b4 --upstream 30 --downstream 120`
  - Initial 3-genome Enterobacteriaceae smoke was intentionally replaced because
    it was too narrow for physical termination biology.
  - Final dataset scope: 24 local non-duplicate GBFF files across
    Enterobacteriaceae, Gram-positive, high-GC, and `data/raw/expanded`.
  - Records: 91,131 hybrid CDS+UTR examples.
  - Packed windows: train 70,650; validation 8,433; test 8,491.
  - Integrity: 0 empty target windows in all splits.
- [x] **Task 1b.5: Verify transfer loading before long MPS training**
  - Dry run loaded 175 same-shaped tensors exactly.
  - Copied 67 shared token rows into `loss_weights`, `tok_emb.weight`, and
    `head.weight`.
  - No missing or unexpected tensors.
  - Config note: `sep_mask_enabled=false` for hybrid tokenization because token id
    3 is `<UNK>`, not `<SEP>`.
- [x] **Task 1b.6: Run the transfer fine-tune**
  - Command:
    `RUN_ID=2026-06-19_physical_termination_transfer_mps_b4_e1 python -m src.codonlm.train_codon_lm --config configs/physical_termination_transfer.yaml`
  - Status: completed 3 epochs.
  - Run dir: `runs/2026-06-19_physical_termination_transfer_mps_b4_e1`.
  - Final checkpoint: `best.pt` and `last.pt` both point to epoch 3.
  - Final validation: `val_loss=4.8822`, `val_next_loss=4.8513`,
    `ppl=127.91`, `val_term_loss=0.3088`.
  - Training trajectory improved every epoch (`val_loss`: 5.496 -> 5.072 -> 4.882).
  - Selected batch optimizer setting: `batch_size=4`, `grad_accum_steps=16`
    on MPS, with observed generation/training throughput around 2.3-2.7 seq/sec.
- [x] **Task 1b.7: Evaluate biological termination**
  - Compare against Stage 2.6, termination-aux, and decoder-biased termination-aux.
  - Required outputs: terminal-stop rate, hard-cap rate, UTR poly-T rate, hairpin score,
    and whether stop behavior appears before or only after length-gated decoder bias.
  - Matched quick prefix evaluation (`--preset quick --seed 1337`) completed for
    Stage 2.6, termination-aux, and physical-transfer checkpoints.
  - Hybrid evaluation support was added to `scripts/eval_generation_prefix.py` so
    it can read `manifest.json` and extract CDS-only prefixes from `hybrid_data.tsv`.
  - Result: physical transfer improved local AA-prefix similarity but did not
    solve natural gene termination.
  - Summary:
    - Stage 2.6 baseline: terminal stop 0%, hard-cap 100%, median GQS 26.62,
      mean AA identity 0.0769.
    - Termination-aux: terminal stop 0%, hard-cap 100%, median GQS 26.44,
      mean AA identity 0.0756.
    - Physical transfer: terminal stop 0%, hard-cap 100%, median GQS 21.40,
      mean AA identity 0.0947.
  - Nonzero termination-bias evaluation (`--termination_stop_bias 8`) did not
    change behavior: the auxiliary head predicted class 4 ("far/no stop") for
    all generated samples, so strict stop bias never activated.
  - Report: `runs/2026-06-19_physical_termination_transfer_mps_b4_e1/scores/physical_termination_eval_report.md`.

## Current Status / Open Failure Modes

- **Solved mechanically:** valid ORF ending can be forced without short-peptide
  collapse using length-gated decoder stop bias from the termination auxiliary
  head.
- **Still unsolved biologically:** generated proteins still do not show improved
  semantic/function/structure quality. The decoder fix does not teach foldability.
- **Completed pilot:** physical-termination transfer run
  `2026-06-19_physical_termination_transfer_mps_b4_e1` completed 3 epochs and
  improved validation loss every epoch.
- **Negative biological result:** hybrid CDS+UTR transfer alone did not produce
  natural stops during prefix generation. The model still hard-caps 100% of
  matched quick samples.
- **Failure mode now identified:** during generation, the termination auxiliary
  head predicts class 4 ("far/no stop") everywhere near the target boundary, so
  strict stop-bias decoding does not activate.
- **Next hypothesis:** off-distribution generated-prefix replay / hard-negative
  training is needed. Train on generated hard-cap failures and label where a
  stop should have occurred, instead of continuing the same in-distribution
  teacher-forced objective.

### Phase 2: Dual-Track Late Fusion (Milestone 2)
- [x] **Task 2.1: Implement Nucleotide Encoder**
  - Created `NucleotideEncoder` module in [biophysics.py](file:///Users/User/github/genomics-lm/src/codonlm/biophysics.py) using a stride-3 1D CNN that downsamples sequence length from $3L \to L$ (averaging nucleotide structure over codon triplets).
  - Pre-trained it on synthetic DNAshape target regressions, dropping MSE loss to **`0.17856`**.
- [x] **Task 2.2: Implement Cross-Attention / Injection**
  - Updated [model_tiny_gpt.py](file:///Users/User/github/genomics-lm/src/codonlm/model_tiny_gpt.py) to map predicted shape vectors (MGW, Roll, EP) to `n_embd` and add them to the generator's representations.
  - Implemented late-fusion **zero-initialization** to ensure pre-trained weights remain completely undisturbed at step 1 of training.
- [x] **Task 2.3: Validate Guidance Performance**
  - Conducted training on 91k hybrid sequences (`2026-07-13_physical_termination_shape_guided_e1_v2`) using a vectorized one-hot lookup table on GPU.
  - Prefix generation validation showed that injecting physical shape embeddings significantly stabilizes gene generation, increasing median GQS by **~25%** (from `21.46` to `26.79` at $k=3$).

### Phase 3: Energy-Based Optimizer (Milestone 3)
- [x] **Task 3.1: Train Bidirectional EBM**
  - Completed training of the upscaled 1024-dim bidirectional EBM (`best_ebm.pt`) to validation loss of **`0.4307`**.
- [x] **Task 3.2: Implement MCMC / Langevin Optimization**
  - Implemented guided sampling blending EBM energy scores with next-token logits to steer generation, which minimized energy to **`-43.33`** (vs baseline `-9.49`) and was integrated into the dashboard playground.
