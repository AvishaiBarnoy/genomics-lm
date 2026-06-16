# Long-Range CodonLM Objectives Plan

## Status

Open. This track depends on the PDB-filtered structural fine-tuning results and
will benefit from the Structural-Aware ProteinCritic labels.

## Tasks

- [ ] Add a config-gated multi-offset LM loss.
- [ ] Add tests for offset target creation and end-of-sequence masking.
- [ ] Run a tiny smoke experiment with offsets `+4`, `+16`, `+32`.
- [ ] Compare validation loss/perplexity against standard next-token training.
- [ ] Add generated-prefix replay or denoising corruption.
- [ ] Evaluate whether recovery training improves termination without shortening genes.
- [ ] Add hooks for structural auxiliary labels from ProteinCritic.
- [ ] Build an offline preference dataset from ESMFold-scored generations.
- [ ] Run a conservative preference-training smoke with KL/replay regularization.

## Biology-Informed Priors

- Helical local structure often involves nearby residues such as `i -> i+3/i+4`.
- Medium/long-range contacts are important for fold topology, beta sheets, and cores.
- Contact order captures average sequence separation between contacting residues; single-domain proteins commonly span a broad relative contact-order range.
- Therefore the objective should not only predict `n+2` or `n+10`; it should combine multi-offset sequence losses with structural/foldability labels.

## Initial Experiments

1. **Offset-only ablation**
   - Base: Stage 2.6 or Stage 3 checkpoint.
   - Data: PDB-filtered subset plus general-CDS replay.
   - Loss: next-token + weighted `+4/+16/+32`.

2. **Denoising/recovery ablation**
   - Corrupt real CDS prefixes.
   - Train model to continue/recover valid CDS.
   - Measure termination, length, and ProteinCritic categories.

3. **Preference smoke**
   - Use ESMFold-scored generated libraries.
   - Prefer higher-pLDDT sequences from the same prompt.
   - Include KL/replay to avoid diversity collapse.

## Metrics

- Original validation perplexity.
- PDB-filtered validation perplexity.
- termination rate and length distribution.
- ProteinCritic stability and protein-type labels.
- mean/max ESMFold pLDDT on matched top-k.
- diversity: pairwise identity and k-mer coverage.
