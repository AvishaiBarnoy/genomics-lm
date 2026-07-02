# Protein Latent Energy-Based Model (EBM) Specification

## Overview
Implement a continuous Latent-Space Energy-Based Model (EBM) and token-monitoring loop detectors for sequence generation in Genomics-LM. By mapping the learned energy of a sequence directly to thermodynamic stability and biological realism, the model will:
1. Transition sequence design from discrete "trial-and-error generation + filtering" to continuous "guided gradient descent optimization" (Langevin dynamics).
2. Integrate early-abort trajectory pruning inside ReD sampling using real-time EBM energy evaluation.
3. Eliminate infinite generation loops with a sliding-window Shannon entropy filter that respects high-GC bacterial sequences.

## Requirements
- **Latent-Space Mapping:** Interface with the pre-trained `ProteinLM` backbone's embedding layer to extract continuous latent representations $z$.
- **Energy Function:** Implement a lightweight neural energy head (MLP or CNN) that inputs embedding $z$ and outputs a scalar energy score $E(z)$ representing stability.
- **NCE Training:** Train the EBM via Noise Contrastive Estimation (NCE) by contrasting high-stability natural sequences (low energy) against perturbed/corrupted mutations and decoys (high energy).
- **Langevin Sampler:** Build an inference sampler that performs continuous Langevin dynamics in the latent space of `ProteinLM`, optimizing embeddings to reach minimum energy before decoding them back to discrete sequence space.
- **Entropy-Based Loop Detection:** Monitor sliding-window Shannon entropy of emitted codon tokens. Detect near-zero entropy loop states to abort generation immediately while permitting complex high-GC bacterial sequences.
- **EBM-Guided ReD Early-Abort:** During step-by-step token generation, evaluate the partial sequence energy. Abort and reset the ReD trajectory if the energy per residue spikes or stays stagnant in a non-biological profile.
- **Hardware Constraint Compatibility:** Ensure the entire forward pass and sampler run with a minimal RAM footprint (<1GB overhead), fully compatible with Apple Silicon M2 local memory limitations.

## Success Criteria
1. **Loop Detection Precision:** The Shannon-entropy loop detector must achieve >99% recall on stuck generator loops while maintaining a 0% false-positive rate on natural high-GC bacterial sequence controls.
2. **Token Efficiency Gain (Early-Abort):** EBM-guided early abort must reduce the average tokens generated per successful sequence in the ReD loop by $\ge 30\%$ compared to the baseline hard-cap policy.
3. **Langevin Optimization Convergence:** Latent Langevin optimization must reduce energy scores on 100% of tested seed sequences, yielding a corresponding increase in predicted ProteinCritic stability scores ($\ge 15\%$ average improvement).
4. **Structural Validity:** Sequence optimization must not degrade translation fidelity (maintaining 100% valid start/stop codon grammar).
