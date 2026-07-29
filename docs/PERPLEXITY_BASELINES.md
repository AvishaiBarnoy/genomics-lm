# Interpreting CodonLM Perplexity

## Calculation

For held-out target codons \(y_1,\ldots,y_N\), negative log-likelihood and
perplexity are:

\[
\mathrm{NLL}=-\frac{1}{N}\sum_{i=1}^{N}\log p(y_i\mid\text{allowed context}),
\qquad
\mathrm{PPL}=e^{\mathrm{NLL}}.
\]

PAD targets are excluded. Lower PPL means the model assigns more probability to
the observed continuation. PPL can be read as an effective average number of
plausible next choices, but it is not literally the number of choices at every
position.

Only the uniform threshold is theoretical from vocabulary size. There are 68
tokens including PAD, so uniform prediction over the 67 evaluable targets gives:

\[
\mathrm{PPL}_{uniform}=67.
\]

The other thresholds are empirical Markov baselines. Their probabilities are
estimated from the frozen training split with additive smoothing and evaluated on
exactly the same held-out tokens as CodonLM:

| Baseline | Prediction | What beating it demonstrates |
| --- | --- | --- |
| Uniform | \(P(x_t)=1/67\) | Learned that targets are not equally frequent |
| Unigram | \(P(x_t)\) | Learned information beyond global token composition |
| Bigram | \(P(x_t\mid x_{t-1})\) | Uses context better than the previous-token table |
| Trigram | \(P(x_t\mid x_{t-2},x_{t-1})\) | Uses information beyond the previous two-token table |

Bigram and trigram histories reset after `<SEP>`, matching the model's segment
boundary. All probabilities are fitted on training data; no validation or test
targets are used to estimate them.

## Current Validation Result

The frozen genome validation comparison uses `3,228,255` non-PAD targets:

| Model | NLL | PPL |
| --- | ---: | ---: |
| Uniform | 4.204693 | 67.000 |
| Unigram | 3.892183 | 49.018 |
| Bigram | 3.782536 | 43.927 |
| Trigram | 3.748532 | 42.459 |
| CodonLM, batch 64, LR `1.5e-4`, seed 1337 | **3.712613** | **40.961** |
| CodonLM, batch 64, LR `1.5e-4`, seed 2027 | **3.724160** | **41.436** |

The selected CodonLM beats trigram in both declared replicates: by `0.035919`
nats/token for seed 1337 (95% packed-window bootstrap CI
`[-0.036887, -0.034984]`) and by `0.024372` for seed 2027 (95% CI
`[-0.025314, -0.023417]`). Its context ablation also improves through 32-128
codons, so the gain is consistent with using information unavailable to a
two-codon Markov table. These are validation results; final test results are
reported only after the configuration is locked.

After locking the configuration, both best checkpoints were evaluated once on the
frozen test split:

| Model | Test NLL | Test PPL | Bits/codon |
| --- | ---: | ---: | ---: |
| Unigram | 3.895213 | 49.167 | 5.6196 |
| Bigram | 3.779984 | 43.815 | 5.4534 |
| Trigram | 3.738549 | 42.037 | 5.3936 |
| CodonLM, seed 1337 | **3.666961** | **39.133** | **5.2903** |
| CodonLM, seed 2027 | **3.676089** | **39.492** | **5.3035** |

Thus the trigram promotion gate holds on the untouched test split for both seeds,
by `0.071588` and `0.062460` nats/token respectively.

## What PPL Says the Model Learned

Crossing these thresholds gives increasingly strong evidence:

1. Below uniform: target-frequency structure.
2. Below unigram: conditional sequence structure, not only codon abundance.
3. Below bigram/trigram: useful context beyond their fixed Markov histories, or a
   better way to generalize those histories.
4. Improvement with longer allowed context: information distributed across longer
   codon spans.

For coding sequence, that information may include amino-acid motifs, synonymous
codon choice, GC context, gene position, taxonomic patterns, translation-related
signals, and other sequence regularities.

PPL alone does **not** prove that the hidden states recover DNA shape, RNA folding,
protein secondary structure, tertiary contacts, function, or fitness. A model can
lower PPL through composition or taxonomy shortcuts. Structural claims therefore
require order-destroying controls, grouped/homology-controlled evaluations,
context ablations, simple sequence baselines, and direct structural targets.

The correct interpretation is:

> Beating trigram establishes that CodonLM learned predictive sequence information
> beyond a two-codon count model. It makes structural learning more plausible and
> worth testing, but does not establish it.
