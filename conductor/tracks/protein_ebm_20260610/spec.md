# Protein Latent Energy-Based Model (EBM) Specification

## Overview
Implement a continuous Latent-Space Energy-Based Model (EBM) for Protein-ML in Genomics-LM. By mapping the learned energy of a sequence directly to thermodynamic stability, the model will transition our sequence design pipeline from discrete "trial-and-error generation + filtering" to continuous "guided gradient descent optimization" (Langevin dynamics).

## Requirements
- **Latent-Space Mapping:** Interface with the pre-trained `ProteinLM` backbone's embedding layer to extract continuous latent representations $z$.
- **Energy Function:** Implement a lightweight neural energy head (MLP or CNN) that inputs embedding $z$ and outputs a scalar energy score $E(z)$.
- **Noise Contrastive Estimation (NCE) Training:** Train the EBM by contrasting high-stability sequences (low energy) from datasets like MegaScale against perturbed/unstable mutations or synthetic negative samples (high energy).
- **Langevin Sampler:** Build an inference sampler that performs continuous Langevin dynamics in the latent space of `ProteinLM`, optimizing embeddings to reach minimum energy before decoding them back to discrete sequence space.
- **Hardware Constraint Compatibility:** Ensure the entire forward pass and sampler run with a minimal RAM footprint (<1GB overhead), fully compatible with Apple Silicon M2 local memory limitations.
