# Corrected CodonLM Training Program

## Status

Planned. Dataset freeze and evaluator validation are prerequisites; no full training
run is authorized until their gates pass.

## Objective

Train a leakage-controlled causal CodonLM from random initialization, establish its
value over simple sequence baselines, and then test long-range, termination, and
biophysical extensions as separately attributable experiments.

This track defines the model sequence within the broader
[leakage-controlled revalidation track](../leakage_controlled_revalidation_20260719/).
That track owns dataset correctness and publication requirements. This track owns
which models are trained, how they are compared, and when an extension is promoted.

## Scientific Questions

1. Does a basic causal Transformer outperform uniform, unigram, bigram, and trigram
   predictors on truly held-out genomes and genera?
2. Do corrected causal embeddings improve controlled downstream tasks?
3. Do multi-offset heads add useful long-range information without degrading the
   primary next-codon model?
4. Does an explicit termination objective improve natural sequence completion rather
   than merely forcing a decoder stop?
5. Does DNA-shape conditioning add information beyond codon identity and local
   nucleotide-context controls?

## Prerequisite Gates

Full training is blocked until all of the following are true:

- The corrected genome-held-out and genus-held-out datasets are immutable and
  identified by versioned manifests and artifact hashes.
- Ambiguity fragmentation, chunking, packing, and vocabulary gates pass; exact
  cross-split CDS copies are absent after recorded quarantine; mandatory protein
  homology reports complete under the declared grouped-holdout policy.
- Every final evaluator consumes explicit frozen artifacts and emits provenance.
- CPU integration and MPS train/save/resume preflights pass.
- The MPS runtime policy is selected by an equal-token quality comparison without
  changing model architecture or objectives.

## Stage 1: Primary Basic CodonLM

The headline corrected model is a decoder-only causal Transformer trained solely on
next-codon prediction.

### Fixed model family

- 10 Transformer layers.
- 8 query heads.
- Embedding width 384.
- Maximum context 512 unless a corrected-data context ablation later promotes a
  shorter context.
- Dropout 0.1, including attention-probability dropout during training.
- Vocabulary resolved exclusively from the frozen tokenizer artifact.
- Standard multi-head or grouped-query attention selected only through the MPS
  runtime/quality gate.

### Excluded from the primary model

- Shape or nucleotide-physics encoder.
- Multi-offset `n+x` prediction heads.
- Termination-distance head or generated-prefix replay.
- Protein critic or energy-guided objective.
- Legacy checkpoint transfer.
- RoPE, SwiGLU, model-size, or other architecture changes mixed into this comparison.

### Required runs

- Genome-held-out training from random initialization at seeds `1337` and `2027`.
- A separately labeled genus-held-out run from random initialization.
- Matched non-PAD token exposure, objective, architecture, tokenizer, and evaluation
  commands across comparable runs.

The two genome seeds form the primary result. Genus holdout measures a harder and
different generalization regime and must not be pooled with genome-holdout metrics.

## Stage 2: Primary Evaluation Gate

Before extensions are trained, the primary model must be compared with uniform,
unigram, bigram, and trigram baselines on the identical held-out token stream. Report
loss, perplexity, bits/codon, token count, and improvement over the best simple
baseline.

Corrected causal embeddings must then be evaluated on EC, essentiality, AMR, and
DNA-shape tasks using their controlled group/homology splits and shared controls.
Generated samples require nucleotide/protein nearest-neighbor and training-match
coverage audits before novelty claims.

Failure to outperform the best simple intrinsic baseline pauses extension training
and triggers a data, objective, optimization, and evaluation audit. Extensions must
not be used to conceal a failed primary model.

## Stage 3: Multi-Offset Long-Range Extension

Multi-offset heads predict future tokens such as `n+4`, `n+8`, `n+16`, and `n+32`
from the causal hidden state. They are auxiliary training signals intended to retain
longer-range information; they do not replace next-token prediction.

Train this as a labeled ablation from a corrected primary checkpoint or under a
matched from-scratch protocol. Use the same frozen data, heldouts, token budget, and
seeds as its primary control. Report each offset loss separately.

Promotion requires:

- Primary next-token validation loss no more than 2% worse than the matched control.
- No termination, stability, memory, or non-finite-update regression.
- Improvement with confidence intervals in at least one predeclared long-range or
  downstream metric.

## Stage 4: Termination and Length Extension

The termination head predicts distance-to-stop categories. Generated-prefix replay
may expose it to off-distribution states encountered during autoregressive decoding.
Decoder stop bias is a separate inference intervention and must not be conflated with
natural model termination.

Compare, using matched prompts and decoding seeds:

1. Primary model with raw or syntax-constrained decoding as declared.
2. Termination-head model without decoder bias.
3. Termination-head model with generated-prefix replay.
4. Decoder-biased generation, reported only as an intervention.

Promotion requires improved natural terminal-stop and hard-cap rates without short
peptide collapse, material perplexity regression, or degradation in sequence-quality
controls.

## Stage 5: Biophysical Shape-Guided Extension

The biophysical extension conditions causal token representations on a nucleotide
shape encoder. It tests whether explicit local physical descriptors add information
beyond sequence identity; its computed stability and shape targets remain proxies,
not experimental evidence.

Run the following matched conditions:

1. Corrected primary CodonLM.
2. Primary CodonLM plus a frozen, provenance-recorded shape encoder.
3. Primary CodonLM plus a jointly trained shape encoder using discriminative learning
   rates.
4. Shape plus termination only after both independent extensions pass their gates.

All extension runs must initialize from a corrected primary checkpoint, or be clearly
labeled as matched from-scratch experiments. Legacy checkpoints cannot support
corrected held-out claims.

Promotion requires paired confidence intervals and improvement over one-hot codon,
random-model, 5-mer, and 7-mer controls under gene/genome-grouped folds. Parameter
movement alone is not evidence of useful co-adaptation.

## Artifact Contract

Every run must preserve:

- Git SHA and exact resolved configuration.
- Source, dataset-manifest, tokenizer, vocabulary, and audit hashes.
- Initialization checkpoint identity or an explicit random-initialization declaration.
- Seed, split regime, model/objective flags, and trainable parameter groups.
- Committed non-PAD tokens, optimizer/scheduler counters, wall time, throughput, peak
  memory, non-finite batches, aborted groups, and termination reason.
- `last` and `best` checkpoints with exact-resume state.
- Machine-readable evaluation results and a human-readable comparison report.

## Claim Discipline

- Legacy results remain historical hypotheses and cannot initialize the primary run.
- Report absolute changes alongside relative ratios.
- Computational stability, Pfam, EC, critic, and DNA-shape scores are proxy outcomes.
- Synonymous generation measures codon choice for a fixed protein, not de novo protein
  discovery.
- Raw, CDS-constrained, decoder-biased, and critic-guided generation are distinct
  protocols.
- A result becomes a corrected headline only after every prerequisite, replication,
  baseline, provenance, and audit gate passes.

## Non-Goals

- Training every implemented extension in one model.
- Selecting an architecture based only on throughput.
- Treating smoke runs as quality evidence.
- Reusing legacy holdout-exposed checkpoints for corrected claims.
- Claiming experimental function or stability from computational proxy scores.
