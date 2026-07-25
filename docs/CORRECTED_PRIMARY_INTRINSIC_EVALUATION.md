# Corrected Primary Intrinsic Evaluation

## Status

Interim seed-1337 result recorded on 2026-07-25. The first corrected primary
genome-holdout model completed training, but it did not pass the predeclared
intrinsic promotion gate. Expensive downstream and generation claims are paused
pending diagnosis.

This is not the final multi-seed primary result. Genome seed 2027 and the separately
labelled genus-holdout run have not yet been trained.

## Checkpoints

| Checkpoint | Epoch | SHA-256 |
| --- | ---: | --- |
| Selected best | 4 | `095327417fdcc72a27473a0ac091356c0c8a6bd12ff4cfcd127c2ccacf9b2a1f` |
| Final | 10 | `1df89cf815624f259f18eaf7e4b2fc49c72e00f212ca4ec09a3a9983881de932` |

Training applied exactly 5,000 optimizer/scheduler steps and 252,384,380 non-PAD
tokens with zero non-finite microbatches, aborted groups, or discarded finite
microbatches. The smoothed validation selector chose epoch 4.

## Frozen Test Results

All rows use the same 2,228,589 non-PAD tokens from the frozen genome-held-out test
artifact. CodonLM perplexity is computed from ordinary unsmoothed cross-entropy;
the label-smoothed training objective is recorded separately.

| Model | NLL (nats/codon) | PPL | Bits/codon |
| --- | ---: | ---: | ---: |
| Uniform | 4.204693 | 67.000 | 6.0661 |
| Unigram | 3.895213 | 49.167 | 5.6196 |
| Bigram | 3.779984 | 43.815 | 5.4534 |
| Trigram | 3.738549 | 42.037 | 5.3936 |
| CodonLM epoch 4 | 3.876740 | 48.267 | 5.593 |
| CodonLM epoch 10 | 3.885411 | 48.687 | 5.605 |

The selected CodonLM beats unigram by 0.01847 nats/codon and 0.90 PPL, but trails
bigram by 0.09676 nats/codon and trigram by 0.13819 nats/codon. Epoch 10 is worse
than epoch 4 on unsmoothed test NLL, so the selected checkpoint remains correct.

## Sequence Controls

The selected epoch-4 checkpoint was evaluated on deterministic controls derived
from the same frozen test artifact:

| Input | NLL (nats/codon) | PPL |
| --- | ---: | ---: |
| Natural | 3.876740 | 48.267 |
| Codons shuffled within each CDS span | 3.876587 | 48.259 |
| Uniform synonymous recoding | 4.217718 | 67.878 |
| Protein-order shuffle plus synonymous recoding | 4.217216 | 67.844 |

The codon-shuffle control preserves each span's exact codon multiset while
destroying order. Its likelihood is effectively identical to natural sequence,
which indicates that the model's current test advantage is dominated by codon
composition rather than sequential ordering. The synonymous control shows strong
sensitivity to native codon usage. The protein-shuffle control is not an isolated
protein-order test because its construction also chooses random synonymous codons.

## Decision

The Phase 3 promotion criterion requires CodonLM to outperform the best simple
intrinsic baseline on identical held-out tokens. Seed 1337 fails that criterion.
Do not use this checkpoint to support corrected downstream or generative claims yet.

Before deciding between retraining and architectural changes:

1. Produce token-class, position, and CDS-span loss decompositions for CodonLM and
   the Markov baselines.
2. Run matched context ablations that retain zero, one, two, and progressively more
   preceding codons.
3. Verify that packing spans and separator masks expose the intended within-CDS
   context during training and evaluation.
4. Quantify paired per-token uncertainty for CodonLM-minus-trigram NLL.
5. Test a predeclared no-label-smoothing or reduced-label-smoothing training
   ablation only if diagnostics show contextual signal is being suppressed rather
   than absent from the data.

Genome seed 2027 should not be treated as a remedy for this effect: replication can
estimate variance, but it is unlikely to close a 0.138-nat/codon gap without a
corrected hypothesis.
