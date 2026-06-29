# Long-Range CodonLM Objectives Spec

## Objective

Teach CodonLM to represent longer-range protein constraints and recover from
off-distribution generation, instead of relying only on next-codon prediction.

## Motivation

Next-token language modeling learns local codon grammar well, and our probes
show it captures DNA electrostatic and 3D shape features. But foldable protein
generation requires longer-range amino-acid constraints: helices, beta-sheet
pairing, hydrophobic cores, membrane topology, domain length, contact order, and
termination placement.

Generation is also off-distribution: during training each prefix comes from a
real biological sequence, while during sampling the model must condition on its
own imperfect previous codons.

## Candidate Objectives

### Multi-Offset Future Prediction

Add auxiliary prediction losses at multiple codon offsets:

- `+2`
- `+4`
- `+8`
- `+16`
- `+32`

This encourages hidden states to encode more than the immediate next codon.
Offsets should be ablated; the initial safe set is `+4`, `+16`, `+32`.

### Denoising / Recovery

Corrupt real CDS prefixes and train the model to recover:

- random synonymous swaps
- random codon masking
- short local codon shuffles
- generated-prefix replay from previous weak models

### Structural Auxiliary Heads

Use translated proteins and external labels to predict:

- protein type from Structural-Aware ProteinCritic
- structure-supported vs not structure-supported
- predicted foldability / pLDDT bucket
- optional contact-order/contact-density labels when PDB-derived labels are available

### Preference Or Reward Training

Construct paired generations from the same prompt:

- preferred: higher ESMFold pLDDT / higher foldability score
- rejected: lower ESMFold pLDDT / disordered soluble candidate

Start with offline preference loss before attempting online ESMFold REINFORCE.

## Acceptance Criteria

- Multi-offset loss can be enabled by config without changing default training.
- Unit tests verify target shifting and masking near sequence ends.
- A small ablation compares baseline next-token vs multi-offset training on validation perplexity and generation metrics.
- Off-distribution recovery training improves termination and reduces obviously invalid continuations.
- Structural auxiliary labels improve at least one downstream generation metric without collapsing diversity.

## Risks

- Multi-offset prediction may improve statistics without improving foldability.
- Strong auxiliary losses may hurt codon dialect or DNAshape representations.
- Reward training can collapse sequence diversity if not KL-regularized.

Mitigation: use replay from general CDS data and evaluate original Stage 2.6 metrics after each experiment.
