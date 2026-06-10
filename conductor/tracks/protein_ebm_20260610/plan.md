# Protein Latent Energy-Based Model (EBM) Implementation Plan

## Phase 1: Architecture & Dataset Extraction
- **Task 1.1: Latent Embedding Extractor**
  - Create a script/utility to extract and cache continuous mean-pooled sequence embeddings from the frozen `ProteinLM` model.
- **Task 1.2: EBM Architecture**
  - Implement `ProteinLatentEBM` in `src/protein_lm/ebm.py` (a small MLP/residual network that accepts embedding vectors and outputs a single scalar energy score).
- **Task 1.3: MegaScale Stability Dataset Loader**
  - Prepare a loader for experimental stability datasets (assigning continuous targets based on fold change/$\Delta \Delta G$).

## Phase 2: NCE Training Loop
- **Task 2.1: Implement training script `src/protein_lm/train_ebm.py`**
  - Build the Noise Contrastive Estimation (NCE) training loop.
  - Implement a corruption/mutation function to generate negative contrastive samples on-the-fly (e.g., amino acid substitution, shuffling, or generation of synthetic decoys).
  - Train the energy head to assign low energy to wild-type/stable sequences and high energy to corrupted decoys.

## Phase 3: Latent Langevin Sampler & Generation Guide
- **Task 3.1: Langevin Sampler**
  - Implement the Langevin dynamics sampling function in `src/protein_lm/sampler.py`:
    $$z_{t+1} = z_t - \eta \nabla_z E(z_t) + \epsilon$$
  - Implement a projection step to project optimized embeddings back to discrete sequence space using the generator's un-embedding projection.
- **Task 3.2: Integration with Model Playground UI**
  - Expose the EBM sampler in the Streamlit web dashboard. Add a visual sequence optimization tab showing the step-by-step energy decrease as the sequence is mutated.

## Phase 4: Unit Testing & Verification
- **Task 4.1: Write unit tests `tests/test_protein_ebm.py`**
  - Test that the energy function outputs lower energy for wild-type sequences than for randomized/corrupted sequences.
  - Verify that the Langevin optimizer converges to a lower energy state.
