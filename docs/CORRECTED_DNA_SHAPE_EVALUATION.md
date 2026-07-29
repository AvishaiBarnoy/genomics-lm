# Corrected DNA-Shape Evaluation

## Question

Does the corrected basic CodonLM make local DNA-shape proxy values linearly
recoverable from its causal codon states?

The 14 targets are computed deterministically from local nucleotide sequence. They
are not independent experimental structure measurements. Consequently, this
benchmark measures representation decodability and shortcut resistance, not proof
that the model learned physical DNA structure.

## Protocol

- Frozen genome-held-out test artifact only.
- Deterministic sampling of 100 windows, balanced 50/50 across the two held-out
  genomes.
- 38,068 codon positions in the primary analysis.
- Two-fold genome grouping: train the Ridge probe on one genome and test on the
  other.
- Five-fold gene grouping reported separately as a less stringent sensitivity
  analysis.
- Identical folds for one-hot codons, centered local 5-mer and 7-mer features,
  random Transformer states, and pretrained states.
- Corrected checkpoints seeds 1337 and 2027.
- Random-model seed 19.
- Final hidden states and layer 2. Layer 2 was motivated independently by the AMR
  train-only experiment and was not selected using DNA-shape results.

## Primary Genome-Grouped Result

Mean R² is averaged across 14 shape properties.

| Representation | Mean R² |
| --- | ---: |
| Centered local 5-mer | **0.767** |
| One-hot codon | 0.445 |
| Random Transformer, final | 0.334 |
| Random Transformer, layer 2 | 0.344 |
| CodonLM seed 2027, layer 2 | -0.640 |
| CodonLM seed 2027, final | -0.660 |
| CodonLM seed 1337, layer 2 | -0.670 |
| CodonLM seed 1337, final | -0.677 |

Negative R² means the fitted probe generalizes worse than predicting the held-out
genome's target mean. With only two independent genomes, confidence intervals are
wide; the near-identical results across checkpoint seeds and layers are therefore
more informative than the nominal two-fold interval.

The centered 7-mer reaches only R² `0.025` in genome transfer, despite its strong
gene-grouped result. This likely reflects sparse 7-mer feature coverage across two
genomes rather than evidence that five bases are universally more informative than
seven.

## Gene-Grouped Sensitivity

| Representation | Mean R² |
| --- | ---: |
| Centered local 5-mer | **0.930** |
| Centered local 7-mer | 0.844 |
| One-hot codon | 0.654 |
| Random Transformer, layer 2 | 0.608 |
| Random Transformer, final | 0.600 |
| CodonLM seed 2027, layer 2 | 0.024 |
| CodonLM seed 1337, layer 2 | 0.017 |
| CodonLM seed 1337, final | 0.010 |
| CodonLM seed 2027, final | 0.001 |

Gene grouping prevents positions from the same gene entering both folds, but genes
from the same genome can appear on both sides. It therefore provides tighter
uncertainty while testing a weaker generalization boundary.

## Conclusion

The corrected basic CodonLM does not linearly expose the computed local DNA-shape
targets. This failure is consistent across two checkpoints, final and early
layers, genome transfer, and gene-grouped sensitivity analysis. Random Transformer
features retain far more locally decodable sequence information.

This does not invalidate the PPL result. It shows that learning longer-range
next-codon dependencies did not organize these local physical proxies in a
linearly accessible form.

A capacity-matched nonlinear probe is still useful as a sensitivity analysis. If
it succeeds, the defensible conclusion would be that shape information is present
but nonlinearly encoded. It would not establish that CodonLM improves on random or
local-sequence controls unless those controls receive the same nonlinear probe and
grouped folds.
