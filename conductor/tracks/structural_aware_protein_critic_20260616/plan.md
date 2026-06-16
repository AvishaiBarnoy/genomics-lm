# Structural-Aware ProteinCritic Plan

## Status

Open. First implementation pass is complete: UniProt protein-type dataset
preparation, dynamic collation, length bucketing, masked pooling, and a starter
config are implemented and tested. Training the first structural-aware critic is
still open.

## Tasks

- [x] Define the first protein-type label schema from UniProt columns.
- [x] Implement `scripts/prepare_protein_type_dataset.py` from local UniProt TSVs.
- [x] Add dynamic length bucketing and a protein collate function.
- [x] Update ProteinCritic models to accept `attention_mask`.
- [x] Replace naive pooling with masked pooling.
- [x] Add multi-label protein-type heads.
- [x] Add tests for label extraction, bucketing, dynamic padding, and masked pooling.
- [ ] Train a first structural-aware ProteinCritic.
- [ ] Add generation-loop reporting for protein type and foldability.
- [ ] Compare generated sequence categories before/after PDB-filtered CodonLM fine-tuning.

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
