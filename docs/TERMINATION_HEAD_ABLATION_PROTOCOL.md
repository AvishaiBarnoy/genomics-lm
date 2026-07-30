# Termination-Head Ablation Protocol

## Hypothesis

The corrected base model assigns termination tokens low but nonzero probability.
Unrestricted decoding recovers some natural stops, but 30-42% of samples still hit
the hard cap. A distance-to-boundary auxiliary task may organize causal hidden
states around CDS endings and improve raw next-token termination after conservative
joint fine-tuning.

The auxiliary head does not directly modify next-token logits. Consequently:

- raw unrestricted generation tests an intrinsic training effect;
- termination-head logit bias is a separate decoder intervention;
- a frozen-backbone head can only help the decoder intervention and is not the
  primary condition.

## Locked Training Condition

- Anchor: corrected seed-1337 promoted checkpoint.
- Frozen genome split and unchanged 68-token vocabulary.
- One epoch, approximately one frozen-corpus exposure.
- Main next-token loss retained.
- Termination loss weight `0.1`.
- EOS (`token 2`) defines the future CDS boundary.
- Distance buckets: exact boundary, 1-3, 4-10, 11-30, and greater than 30/no future
  boundary in the packed segment.
- No replay, multi-offset heads, shape encoder, critic, termination bias, or forced
  stop during training.
- Joint update: backbone learning rate `5e-6`; new termination head learning rate
  `1e-4`.

## Class Balance

The frozen training labels contain 25,238,438 valid positions:

| Class | Count | Fraction | Weight |
| --- | ---: | ---: | ---: |
| exact boundary | 74,659 | 0.296% | 12.364329 |
| 1-3 | 223,561 | 0.886% | 7.145187 |
| 4-10 | 519,672 | 2.059% | 4.686482 |
| 11-30 | 1,471,522 | 5.830% | 2.785020 |
| far/no future boundary | 22,949,024 | 90.929% | 0.705228 |

Weights are square-root inverse frequency normalized to unit expected weight.
This prevents the auxiliary head from learning only the majority class without
using the extreme weights produced by full inverse frequency.

## Evaluation

Compare the anchor and head-only checkpoint with identical held-out prompts and
sample seeds:

1. raw unrestricted sampling, temperature `1.0`;
2. syntax-constrained sampling, reported separately;
3. termination-head-biased decoding, reported only as an inference intervention.

Report:

- frozen validation and test next-token NLL/PPL;
- termination loss and per-class accuracy/confusion;
- natural-stop, EOS, early-stop, and hard-cap rates;
- generated and held-out length distributions;
- GC and codon-distribution divergence;
- diversity, exact coverage, and nucleotide/protein nearest neighbors;
- runtime, memory, and nonfinite-update counts.

## Promotion Gate

- Test next-token NLL may regress by no more than 2% relative to the anchor.
- Raw unrestricted hard-cap rate must decrease with no short-sequence collapse.
- Any decoder-biased gain must remain labelled as an inference intervention.
- Replay is authorized only if the head-only condition remains inadequate.
