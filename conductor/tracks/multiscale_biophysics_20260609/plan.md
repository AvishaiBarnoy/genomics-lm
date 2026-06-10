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

### Phase 2: Dual-Track Late Fusion (Milestone 2)
- [ ] **Task 2.1: Implement Nucleotide Encoder**
  - Create a lightweight `NucleotideEncoder` module in PyTorch (e.g., 2-layer CNN or 1-layer local attention transformer).
  - Train it to predict local DNAshape features (MGW, Roll, EP) on sliding 60 bp windows.
- [ ] **Task 2.2: Implement Cross-Attention / Injection**
  - Modify `CausalSelfAttention` in `model_tiny_gpt.py` to optionally accept the encoder's physical embeddings.
  - Implement caching for the encoder's representations during autoregressive generation to minimize inference overhead.
- [ ] **Task 2.3: Validate Guidance Performance**
  - Run a prefix generation benchmark and verify that stop codon and terminator transition probabilities are guided by the encoder.

### Phase 3: Energy-Based Optimizer (Milestone 3)
- [ ] **Task 3.1: Train Bidirectional EBM**
  - Train a bidirectional network that outputs a scalar "energy" value for a nucleotide sequence.
  - Train it on ground-truth genomic sequences (low energy) vs mutated/shuffled sequences (high energy).
- [ ] **Task 3.2: Implement MCMC / Langevin Optimization**
  - Build a sampler that starts from a translated codon sequence, randomly swaps synonymous codons, and uses the EBM energy gradient to choose the thermodynamically optimal synonymous mRNA sequence.
