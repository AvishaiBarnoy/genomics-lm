# PDB-Filtered Structural Fine-Tuning Spec

## Objective

Create the first direct structural training signal for CodonLM by fine-tuning the best Stage 2.6 checkpoint on coding sequences whose translated proteins have experimental structure evidence or high-confidence structural annotations.

## Motivation

The structured generation track showed that sampling-time filters can improve ProteinCritic stability, but they did not improve ESMFold pLDDT. That means the generator itself has not learned a strong preference for foldable protein sequences. A PDB-filtered fine-tune changes the training distribution rather than only filtering samples after generation.

## Inputs

- Source CDS DNA: `data/processed/stage2.6_large_master_dna.txt`
- Source CDS metadata: `data/processed/stage2.6_large_master_meta.tsv`
- Structure annotations:
  - Current supported path: curated `line_idx` list.
  - Future path: enriched CDS metadata with `protein_id`, `locus_tag`, or `gene` fields joined to UniProt/PDB annotations.

## Outputs

- Filtered CDS subset: `data/processed/structured_pdb/cds_dna.txt`
- Filtered metadata: `data/processed/structured_pdb/cds_meta.tsv`
- Tokenized/packed train/val/test NPZs under `data/processed/structured_pdb_pack/`
- Fine-tuned checkpoint under `runs/stage3_structured_pdb_finetune/`
- Evaluation report comparing baseline Stage 2.6 vs structured fine-tune on:
  - stop behavior
  - ProteinCritic stability
  - ProteinCritic family confidence
  - ESMFold pLDDT on matched generation budgets

## Current Constraint

The existing Stage 2.6 metadata contains only `line_idx` and `genome`. It cannot be automatically joined to UniProt/PDB because protein-level identifiers were not preserved during CDS extraction. The first implementation therefore supports curated source line indices and records the missing metadata requirement explicitly.
