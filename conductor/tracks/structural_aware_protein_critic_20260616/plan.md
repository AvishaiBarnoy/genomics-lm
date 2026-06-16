# Structural-Aware ProteinCritic Plan

## Status

Open. This is the next critic-side track after PDB-filtered structural fine-tuning.

## Tasks

- [ ] Define the first protein-type label schema from UniProt columns.
- [ ] Implement `scripts/prepare_protein_type_dataset.py` from local UniProt TSVs.
- [ ] Add dynamic length bucketing and a protein collate function.
- [ ] Update ProteinCritic models to accept `attention_mask`.
- [ ] Replace naive pooling with masked pooling.
- [ ] Add multi-label protein-type heads.
- [ ] Add tests for label extraction, bucketing, dynamic padding, and masked pooling.
- [ ] Train a first structural-aware ProteinCritic.
- [ ] Add generation-loop reporting for protein type and foldability.
- [ ] Compare generated sequence categories before/after PDB-filtered CodonLM fine-tuning.

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
