# Structural-Aware ProteinCritic Spec

## Objective

Extend ProteinCritic from a family/function/stability scorer into a structural
triage model that can distinguish soluble folded proteins, membrane proteins,
signal peptides, low-complexity/disordered proteins, short peptides, and
structure-supported proteins.

## Motivation

The current ProteinCritic can provide useful stability and family signals, but
generated CodonLM sequences still score low under ESMFold and often look novel
or disordered. Low pLDDT is not always equivalent to a bad biological product:
membrane helices, signal peptides, short peptides, and intrinsically disordered
regions can all be biologically real while scoring poorly as compact soluble
folds.

Before using ProteinCritic as a stronger training or filtering signal, it needs
to answer a more basic question: what kind of protein is this sequence expected
to encode?

## Proposed Labels

Initial multi-label targets should be derived from local UniProt columns:

- `structured_pdb`: UniProt row has `3D-structure` or PDB evidence.
- `membrane`: keywords/features/location mention membrane or transmembrane.
- `signal_secreted`: features/location mention signal peptide, secreted, or periplasmic.
- `disordered_low_complexity`: features mention region, repeat, low complexity, coiled coil, or disorder-related annotations.
- `enzyme`: EC number is present.
- `short_peptide`: amino-acid length below a chosen threshold, initially `< 50 aa`.
- `soluble_candidate`: no membrane/signal/disorder flag and length is in a normal folded-domain range.

These should be multi-label, not mutually exclusive. A protein can be both a
membrane protein and an enzyme.

## Model/Data Changes

- Add a dataset builder that converts UniProt rows into multi-label targets.
- Add ProteinCritic heads for protein type and foldability.
- Add dynamic collation and length bucketing for protein batches.
- Use masked pooling so padding does not contaminate sequence embeddings.
- Preserve existing Pfam, EC, and stability heads.

## Acceptance Criteria

- ProteinCritic can train with variable-length batches and an attention mask.
- Unit tests cover dynamic collation, masked pooling, and multi-label target creation.
- A first classifier report includes AUROC/F1 for membrane, signal/secreted, enzyme, short-peptide, and structure-supported labels.
- The generation loop can report protein type predictions alongside family, EC, and stability.

## Downstream Use

This track feeds the next generation-control stage:

- Reject “bad fold” only when the target class is expected to be soluble/folded.
- Route membrane or peptide-like generations to different evaluation criteria.
- Create foldability labels for CodonLM preference/reward training.
