# Corrected Primary Intrinsic Evaluation

For checkpoint or hyperparameter selection, evaluate the frozen validation split
without touching the final test split:

```bash
python -m scripts.evaluate_test \
  --run_dir runs/<run-id> \
  --split validation \
  --manifest data/processed/corrected/corrected-codonlm-v1/genome/manifest.json
```

This records ordinary unsmoothed NLL and perplexity under `validation_*` keys and
binds the selected file to the manifest's `val_tokens` artifact.

## Status

The locked batch-64, LR `1.5e-4` basic configuration passed the frozen-test
trigram gate in both seeds: PPL `39.133` (1337) and `39.492` (2027), versus
trigram `42.037`. See [Perplexity Baselines](PERPLEXITY_BASELINES.md) and the
[replication record](benchmarks/corrected_lr15_replication.json) for the current
results and checkpoint identities.

The single-seed evaluation below is retained as historical evidence for the earlier
training configuration; its failed promotion decision was superseded by the locked
replicated configuration.

## Reproducible multi-seed check-up

The command below fits unigram, bigram, and separator-aware trigram baselines once
from the manifest's training tokens, then evaluates each explicitly selected
checkpoint against that same frozen test artifact:

```bash
python -m scripts.evaluate_primary_models \
  --manifest data/processed/corrected/corrected-codonlm-v1/genome/manifest.json \
  --checkpoint seed1337=runs/corrected-codonlm-v1-batch64-lr-ablation-lr_1_5e4/checkpoints/best.pt \
  --checkpoint seed2027=runs/corrected-codonlm-v1-batch64-lr15-seed2027/checkpoints/best.pt \
  --output-dir reports/corrected-primary-eval-2026-10-05
```

The output directory must be new. It contains a combined Markdown/JSON
report and isolated benchmark receipts/results for each checkpoint. The report
records the manifest, train/test/vocabulary and checkpoint hashes, evaluator
revision, baseline settings, token-count agreement, and whether every input stayed
unchanged through evaluation. Evaluation runs on CPU. Checkpoint selection remains
explicit; the script does not infer `best.pt` versus `last.pt` for the user.

To run one seed, pass one `--checkpoint NAME=PATH`. Add another only when its
checkpoint is part of the already-authorized evaluation set; this command does not
train or select new seeds.

## Historical interim seed-1337 evaluation

### Checkpoints

| Checkpoint | Epoch | SHA-256 |
| --- | ---: | --- |
| Selected best | 4 | `095327417fdcc72a27473a0ac091356c0c8a6bd12ff4cfcd127c2ccacf9b2a1f` |
| Final | 10 | `1df89cf815624f259f18eaf7e4b2fc49c72e00f212ca4ec09a3a9983881de932` |

Training applied exactly 5,000 optimizer/scheduler steps and 252,384,380 non-PAD
tokens with zero non-finite microbatches, aborted groups, or discarded finite
microbatches. The smoothed validation selector chose epoch 4.

### Frozen Test Results

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

### Sequence Controls

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

### Context Diagnosis

The manifest-bound context sweep passed the independently reconstructed causal and
separator-mask audit. Resetting trigram history after `<SEP>` did not change its
aggregate result to six decimals, so the original cross-boundary trigram defect was
real but too rare to explain the baseline gap.

| Input attention window | NLL | PPL |
| ---: | ---: | ---: |
| 1 | 3.938233 | 51.328 |
| 2 | 3.880021 | 48.425 |
| 4 | 3.876767 | 48.268 |
| 8 | 3.876762 | 48.268 |
| 32 | 3.876746 | 48.267 |
| 128 | 3.876747 | 48.267 |
| Full | 3.876740 | 48.267 |

The checkpoint uses short context: increasing the window from one to two input
tokens improves PPL by 2.90, and four tokens capture essentially the entire
full-context result. There is no measurable benefit beyond four input tokens.
Despite using local context, CodonLM remains worse than the explicit bigram and
trigram estimators.

The paired CodonLM-minus-trigram difference is `+0.138191` nats/token with a 95%
packed-window bootstrap interval of `[+0.136469, +0.139874]`. This excludes run
noise as an explanation.

Boundary losses are disproportionately high:

- prediction immediately after `<SEP>`: PPL `993.5` over 2,160 tokens;
- all segment starts: PPL `107.6` over 7,872 tokens;
- stop codons: PPL `473.9` over 8,387 tokens;
- ordinary codons: PPL `47.36` over 2,159,052 tokens.

Those rare boundary classes do not account for the full baseline gap, but they
identify a separate grammar weakness. Windows containing chunk continuations are
not worse than other windows (`47.42` versus `48.87` PPL), so chunk continuation is
not the immediate failure mode.

### Historical Decision

The Phase 3 promotion criterion requires CodonLM to outperform the best simple
intrinsic baseline on identical held-out tokens. Seed 1337 fails that criterion.
Do not use this checkpoint to support corrected downstream or generative claims yet.

The mask, context, decomposition, and paired-uncertainty diagnostics are complete.
They show connected but underused local context rather than a disconnected
Transformer. The next experiment is the predeclared matched regularization matrix:
current regularization, no label smoothing, no smoothing plus lower dropout, and
the same condition with untied embeddings.

Do not add a convolutional branch, `n+x` heads, or another architecture extension
until that matrix determines whether ordinary next-token optimization can beat the
count baselines.

Genome seed 2027 should not be treated as a remedy for this effect: replication can
estimate variance, but it is unlikely to close a 0.138-nat/codon gap without a
corrected hypothesis.
