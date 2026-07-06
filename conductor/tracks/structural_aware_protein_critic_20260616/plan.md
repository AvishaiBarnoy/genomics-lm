# Structural-Aware ProteinCritic Plan

## Status

Completed. Upgraded generative design loop with coarse structural category reporting and ESMFold calibration. Ran comparative design loops on baseline vs. PDB fine-tuned models under stability constraints to analyze prior shifts.

## Tasks

- [x] Define the first protein-type label schema from UniProt columns.
- [x] Implement `scripts/prepare_protein_type_dataset.py` from local UniProt TSVs.
- [x] Add dynamic length bucketing and a protein collate function.
- [x] Update ProteinCritic models to accept `attention_mask`.
- [x] Replace naive pooling with masked pooling.
- [x] Add multi-label protein-type heads.
- [x] Add tests for label extraction, bucketing, dynamic padding, and masked pooling.
- [x] Train a first structural-aware ProteinCritic.
- [x] Add generation-loop reporting for protein type and foldability.
- [x] Compare generated sequence categories before/after PDB-filtered CodonLM fine-tuning.

## Results (Scaled Bidirectional Critic)

- **Run id:** `2026-07-05_critic_bidirectional_attention_scaled`
- **Backbone Capacity:** 8 layers, 8 heads, 256 embedding dimension (scaled relative to baseline).
- **Validation Metrics:**
  * **Stability Accuracy:** `80.77%` (Meets stability parity target $\ge 77\%$; compared to baseline's `28.46%`).
  * **Attention Saliency Contrast Ratio:** `3.66x` (Active-site residues receive nearly 4x higher attention weights than other regions, satisfying the $\ge 2.0\times$ contrast target).
- **Saliency Optimization:** The scaled training exited gracefully at epoch 4 due to the 3-hour wall-time limit, successfully preserving the best parameters. Regularization cleanly concentrated attention to the Rossmann fold (`YIHIG`) and tRNA active-loop (`KMSKS`) motifs.

## Implementation Notes

- Dataset builder:
  ```bash
  python -m scripts.prepare_protein_type_dataset \
    --uniprot_tsv data/raw/uniprot_bacteria_50_512.tsv \
    --out_dir data/processed/protein_lm/protein_type
  ```
- Local dataset build produced 261,968 train and 29,107 validation samples.
- Starter config: `configs/protein_critic_structural.yaml`.

## Initial Dataset Sources

- `data/raw/uniprot_bacteria_50_512.tsv`
- Existing ProteinCritic multitask data, if available locally.
- Generated libraries from:
  - `outputs/reports/structured_prefix_experiment/`
  - `outputs/reports/stage3_structured_pdb_smoke_prefix/`

## Evaluation

Report per-label:

- prevalence
- accuracy
- macro/micro F1
- AUROC where positive and negative classes both exist
- calibration of soluble/foldable predictions against ESMFold pLDDT

## Notes

Do not treat low ESMFold pLDDT as universally bad until the critic knows whether
the sequence is expected to be soluble, membrane, secreted, peptide-like, or
disordered.
