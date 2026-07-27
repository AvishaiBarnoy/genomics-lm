# Context Learning Diagnostics

## Markov Comparisons

A Markov baseline predicts the next token from a deliberately limited history:

- unigram: no preceding token, only global target frequencies;
- bigram: the current input token, which is the target's immediately preceding token;
- trigram: the target's two immediately preceding tokens.

These models establish whether a Transformer provides value beyond short transition
tables. They are fitted only on the training split and evaluated on the same
manifest-bound held-out tokens as CodonLM.

Markov history resets after `<SEP>` to match CodonLM's attention contract. In
particular, the trigram must not use a token from the preceding CDS when predicting
the first token after `<SEP>`.

## Diagnostic Command

```bash
caffeinate -i python -m scripts.diagnose_context_learning \
  --run-dir runs/corrected-codonlm-v1-genome-seed1337 \
  --checkpoint-name best.pt \
  --train data/processed/corrected/corrected-codonlm-v1/genome/train_bs512.npz \
  --test data/processed/corrected/corrected-codonlm-v1/genome/test_bs512.npz \
  --manifest data/processed/corrected/corrected-codonlm-v1/genome/manifest.json \
  --itos data/processed/corrected/corrected-codonlm-v1/genome/itos.txt \
  --packing-tsv data/processed/corrected/corrected-codonlm-v1/genome/test_packing.tsv \
  --context-windows 1,2,4,8,32,128,full \
  --batch-size 4 \
  --device mps \
  --output-prefix runs/corrected-codonlm-v1-genome-seed1337/diagnostics/context_learning
```

An attention window includes the current input token:

| Window | Target history available |
| ---: | --- |
| 1 | immediately preceding target token; bigram-like |
| 2 | two preceding target tokens; trigram-like |
| 4+ | progressively longer local history |
| full | complete causal history within the current segment |

The command:

- verifies the causal and separator mask against an independently constructed mask;
- reruns segment-aware uniform/unigram/bigram/trigram baselines;
- measures checkpoint NLL at each context window;
- decomposes full-context loss by target class, segment position, boundary, and
  windows containing chunk continuations;
- bootstraps the paired CodonLM-minus-trigram NLL difference over packed windows;
- binds the checkpoint, vocabulary, manifest, token arrays, and packing metadata.

## Regularization Ablation

Materialize the predeclared matched configs:

```bash
python -m scripts.materialize_regularization_ablation \
  --matrix configs/corrected_regularization_ablation.yaml \
  --output-dir runs/corrected-regularization-ablation/configs
```

The matrix contains four two-epoch, random-initialized conditions:

1. current label smoothing, dropout, and tied embeddings;
2. no label smoothing;
3. no label smoothing with dropout reduced to 0.05;
4. no label smoothing, dropout 0.05, and untied input/output embeddings.

All other data, architecture, optimizer, seed, exposure, and scheduler fields remain
identical. These runs are diagnostics, not primary replicates, and their generated
configs intentionally omit `primary_training_contract`.

Do not run the matrix until the context and mask diagnostic completes. A masking or
packing defect invalidates a regularization comparison.

### Result

All four variants completed 1,000 optimizer steps and `50,476,876` non-PAD
training tokens from random initialization. Manifest-bound unsmoothed evaluation
used the same `3,228,255` validation targets:

| Variant | Validation NLL | Validation PPL |
| --- | ---: | ---: |
| Reference | 3.895214 | 49.167 |
| No smoothing | 3.891479 | 48.983 |
| No smoothing, dropout 0.05 | 3.910927 | 49.945 |
| No smoothing, dropout 0.05, untied embeddings | 3.811323 | 45.210 |
| Bigram baseline | 3.782536 | 43.927 |
| Trigram baseline | 3.748532 | 42.459 |

Untying the input and output embeddings produced the only substantial improvement,
reducing PPL by about 8% relative to the reference. Removing smoothing alone was
nearly neutral, and lower dropout with tied embeddings was worse. The selected
diagnostic variant still trails both Markov baselines, so it is an optimization
candidate rather than a promoted primary model. Exact checkpoint and dataset hashes
are recorded in `docs/benchmarks/corrected_regularization_ablation.json`.

## Local Convolution Versus `n+x`

A causal convolutional branch summarizes recent input tokens before the ordinary
next-token prediction:

```text
recent past -> local convolution -> n+1 logits
```

The earlier `n+x` heads use a shared causal representation to predict more distant
future targets:

```text
past -> Transformer -> separate n+4, n+8, ... heads
```

They are complementary but not equivalent. A local convolution supplies an
inductive bias for short motifs and transitions. `n+x` objectives provide auxiliary
supervision about longer-range future tokens. Adding `n+x` cannot repair a model
that has not demonstrated bigram-level contextual learning, so it remains gated
until the basic next-token diagnostic passes.
