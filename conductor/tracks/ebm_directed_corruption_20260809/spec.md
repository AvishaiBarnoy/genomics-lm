# Directed ProteinEBM Corruption Specification

## Objective

Replace uniform-only amino-acid substitution with an explicitly declared and
measured mixture of negative distributions. The resulting ProteinEBM remains a
natural-versus-defined-corruption ranker; energy must not be presented as
experimental stability, function, or fitness without independent validation.

## Requirements

- Preserve the current 20% uniform substitution process as a frozen baseline.
- Fail fast unless source and decoy sequences use the declared 20-residue alphabet.
- Implement independently selectable corruption families:
  - uniform substitutions;
  - BLOSUM-conservative substitutions;
  - Grantham-radical physicochemical substitutions;
  - ProteinLM-low-likelihood contextual substitutions.
- Record corruption family, mutated positions, source/replacement residues, and
  severity for every generated decoy used in evaluation.
- Prevent identity replacements from counting as mutations.
- Keep corruption construction task-owned and deterministic under checkpoint resume.
- Make mixture proportions and mutation-count/rate policies validated configuration.
- Report loss, ranking accuracy, energy gap, and calibration separately for every
  corruption family and severity band, plus the declared mixture aggregate.
- Keep train, validation, and test corruption seeds and artifacts reproducible.

## Scientific Controls

- Hold the ProteinCritic checkpoint, natural-sequence split, EBM architecture,
  optimizer budget, and seed set constant across corruption ablations.
- Select a mixture using validation only and open the test split once after the
  decision is recorded.
- Compare against uniform-only, untrained-head, and simple physicochemical scoring
  baselines.
- Audit whether corruption families are trivially separable by amino-acid
  composition, sequence length, charge, or hydrophobicity before attributing gains
  to contextual protein understanding.
- Treat structure-aware corruption as a later extension requiring trustworthy
  residue annotations or structures; do not infer burial, interfaces, or active
  sites from hand-written sequence rules.

## Success Criteria

1. Every decoy is reproducible and contains only the declared amino-acid alphabet.
2. The selected mixture improves conservative/contextual held-out ranking without
   losing broad uniform-negative discrimination.
3. Performance is not explained solely by simple composition or charge baselines.
4. Energy ordering is tested against an external mutation-effect or stability set
   before energy is used as a biological quality claim.

## Non-Goals

- Changing ProteinCritic or CodonLM parameters.
- Using the final test split to tune corruption proportions.
- Claiming that every synthetic mutation is deleterious.
- Combining this scientific objective change with training-engine migration.
