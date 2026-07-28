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

## Effective-Batch Ablation

The selected regularization condition fixes:

```yaml
label_smoothing: 0.0
dropout: 0.05
tie_embeddings: false
batch_size: 4
```

The completed effective-batch-128 condition is reused as the anchor. Two additional
random-initialized conditions change only accumulation and the token-aligned
scheduler horizon:

| Effective batch | Accumulation | Optimizer steps over two epochs |
| ---: | ---: | ---: |
| 128 | 32 | 1,000 |
| 64 | 16 | 2,000 |
| 32 | 8 | 4,000 |

Every condition uses seed 1337 and processes `50,476,876` non-PAD tokens. Physical
batch size remains four, so device activation memory is approximately matched.
This is a token- and compute-exposure comparison: smaller effective batches receive
more frequent, noisier optimizer updates. It does not establish that a smaller
batch has a universally better asymptotic optimum.

Selection uses manifest-bound unsmoothed validation NLL. The frozen test split
remains unavailable. A useful result must improve on the batch-128 PPL of `45.210`;
the primary promotion gate additionally requires beating validation bigram
(`43.927`) and trigram (`42.459`).

### Result

All conditions completed their declared optimizer steps and exactly `50,476,876`
non-PAD training tokens without nonfinite microbatches, aborted accumulation groups,
or discarded finite work.

| Effective batch | Best epoch | Validation NLL | Validation PPL |
| ---: | ---: | ---: | ---: |
| 128 | 2 | 3.811323 | 45.210 |
| 64 | 1 | 3.763812 | 43.112 |
| 32 | 2 | 3.886750 | 48.752 |
| Bigram | - | 3.782536 | 43.927 |
| Trigram | - | 3.748532 | 42.459 |

Batch 64 reduces PPL by 4.64% relative to batch 128 and beats bigram by `0.018723`
nats/token. It still trails trigram by `0.015280` nats/token, so the primary gate
remains failed. Batch 32 is substantially worse. The batch-64 checkpoint selected
epoch 1; epoch 2 regressed to PPL `44.305`.

The result rejects a monotonic "more updates is better" explanation. Because the
learning rate remained `3e-4` while batch size decreased, gradient noise and update
magnitude changed together. The batch-32 degradation and batch-64 epoch-2 regression
justify a narrow learning-rate ablation at effective batch 64 before introducing a
new architecture. Exact results and checkpoint hashes are recorded in
`docs/benchmarks/corrected_effective_batch_ablation.json`.

### Selected-checkpoint context diagnosis

The batch-64 winner changes the earlier context conclusion:

| Input attention window | Validation PPL |
| ---: | ---: |
| 1 | 78.474 |
| 2 | 62.773 |
| 4 | 53.635 |
| 8 | 48.411 |
| 16 | 45.452 |
| 32 | 43.912 |
| 64 | 43.344 |
| 128 | 43.189 |
| Full | 43.112 |

The original tied checkpoint saturated by four input tokens. The selected untied
batch-64 checkpoint instead gains strongly through 32 codons, continues improving
through 64-128, and retains a small full-context gain. It therefore demonstrates
genuine longer-context use, although it still trails trigram. The paired
CodonLM-minus-trigram deficit is `+0.015280` nats/token with a 95% packed-window
bootstrap interval of `[+0.014204, +0.016337]`, so the remaining gap is small but
statistically robust.

Chunk continuations are not the failure source: their PPL is `42.808`, compared
with `43.295` for other windows. Stop-codon prediction remains a distinct weakness
at PPL `484.316`. These results weaken the case for immediately adding a purely
local convolution. Finish the batch-64 learning-rate ablation first; if the trigram
gap remains, use transition-level error analysis to decide between a local residual,
amino-acid/codon factorization, and other targeted interventions.

## Candidate Architecture Interventions

Architecture changes remain gated on the batch-64 learning-rate result. The
selected model now uses 32-128 codons of context, so interventions must preserve
that gain. The proposed order is:

1. Add a zero-initialized causal convolutional residual with local kernels such as
   3, 5, and 9 codons. This directly represents short transitions while preserving
   the initial Transformer function.
2. Evaluate a separately labelled Markov-residual hybrid that adds a learned
   Transformer correction to fixed trigram logits. This is useful for practical PPL
   but cannot demonstrate that the Transformer independently learned trigram
   structure.
3. Factor next-codon probability into next-amino-acid and synonymous-codon terms.
   Report both components so protein-sequence prediction is not conflated with
   organism-specific synonymous choice.
4. Consider multi-scale local/global attention and segment-relative position resets
   only after the narrower local intervention is measured.

Multi-offset `n+x` heads supervise distant targets and are not a substitute for
learning the ordinary next-token transition. Width, depth, RoPE, and SwiGLU do not
specifically target the observed failure and remain lower priority.

## Biological Context Length

There is no single biological correlation length. The trigram estimator models
`P(x_t | x_(t-2), x_(t-1))`: two preceding codons, or six preceding nucleotides,
condition the next codon. This is a minimal test of context-dependent sequence
grammar, not a structural model.

Relevant scales differ by mechanism:

| Mechanism | Representative scale |
| --- | --- |
| Codon-pair preference | Two adjacent codons |
| Local DNA shape | A sliding pentamer, about 5 bp |
| Ribosome-protected mRNA | Roughly 20-30 nt, about 7-10 codons |
| Local protein secondary-structure prediction | Commonly local windows around 13-17 residues |
| Protein tertiary contacts | Often tens to hundreds of residues apart |
| RNA secondary structure | Can pair positions tens to hundreds of nucleotides apart |

Primary studies support non-random adjacent codon preferences and context-dependent
translation speed ([Buchan et al., 2006](https://pmc.ncbi.nlm.nih.gov/articles/PMC1363775/);
[Chevance et al., 2014](https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1004392)).
The DNAshape method predicts local shape from sliding pentamers
([Zhou et al., 2013](https://pmc.ncbi.nlm.nih.gov/articles/PMC3692085/)).
Ribosome profiling observes protected fragments around 20-30 nt, but footprint
length is not itself a statistical correlation length
([Lareau et al., 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4052883/)).
Protein secondary structure contains short-range information, while complete folds
depend strongly on nonlocal contacts
([Crooks and Brenner, 2004](https://academic.oup.com/bioinformatics/article/20/10/1603/237316);
[Adhikari et al., 2017](https://link.springer.com/article/10.1186/s12859-017-1807-5)).

Lower PPL is therefore necessary for demonstrating improved sequence modeling but
is not evidence of structural recovery by itself. The structural hypothesis must be
tested through order-destroying sequence controls, context ablations, grouped
DNA-shape baselines, homology-controlled protein evaluations, and explicit controls
for amino-acid, codon, GC, and taxonomic composition.

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
