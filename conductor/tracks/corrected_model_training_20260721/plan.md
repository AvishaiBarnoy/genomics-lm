# Corrected CodonLM Training Program Plan

## Status

In progress. The corrected datasets have passed the local freeze; the program remains
blocked on evaluator fixtures, immutable training configs, and the MPS policy gate.

## Phase 0: Close Pre-Training Gates

- [x] Pin the intended 24-source GBFF inventory by explicit assembly ID, byte size,
  and SHA-256 digest.
- [x] Add a fail-closed command that builds genome/genus protocols, validates their
  shared source/vocabulary/packing contract, and emits a content-addressed freeze
  index.
- [x] Complete the local Phase 4 build of immutable
  genome-held-out and genus-held-out datasets.
- [x] Record source, split, fragment, chunk, packed-array, vocabulary, and audit
  hashes plus count summaries.
- [ ] Complete frozen-manifest evaluator fixtures and provenance validation.
- [ ] Resolve generation protocol issue #85 before generation comparisons.
- [ ] Benchmark issue #90 on MPS and either promote batch-aware memmap conversion or
  close it with evidence of no useful benefit.
- [ ] Select the MPS runtime policy through an equal-token quality gate.
- [ ] Add immutable corrected primary configs and config-contract tests.

Exit gate: the frozen artifacts reproduce, all leakage gates pass, evaluator fixtures
pass, MPS preflight passes, and no unresolved semantic choice can change the training
stream.

## Phase 1: Train the Primary Basic Model

- [ ] Run a bounded end-to-end pilot from random initialization on the frozen
  genome-held-out dataset; verify loss scale, token counters, validation, checkpoint,
  resume, memory, and wall-time estimates.
- [ ] Train the genome-held-out primary model at seed `1337`.
- [ ] Train the identical genome-held-out primary model at seed `2027`.
- [ ] Train the separately labeled genus-held-out primary model from random
  initialization.
- [ ] Verify matched non-PAD exposure and config identity across comparable runs.
- [ ] Archive complete run manifests, checkpoints, logs, and failure telemetry.

Exit gate: all declared primary runs complete without leakage, OOM, non-finite update,
counter mismatch, or provenance failure.

## Phase 2: Evaluate and Decide on the Primary Model

- [ ] Evaluate uniform, unigram, bigram, trigram, and CodonLM on identical test tokens.
- [ ] Report per-seed and aggregate loss, perplexity, bits/codon, and improvement over
  the best simple baseline.
- [ ] Extract causal embeddings with dataset/checkpoint/vocabulary/code provenance.
- [ ] Run EC, essentiality, AMR, and DNA-shape evaluations with controlled splits and
  shared controls.
- [ ] Run generated-sequence memorization, nucleotide-neighbor, and protein-neighbor
  audits using the separated generation protocols.
- [ ] Publish the corrected primary report with confidence intervals and limitations.
- [ ] Record a go/no-go decision for extension training.

Exit gate: the primary model outperforms the best simple intrinsic baseline and the
report satisfies the corrected promotion criteria. Otherwise pause and audit.

## Phase 3: Multi-Offset `n+x` Ablation

- [ ] Predeclare offsets, weights, initialization policy, token budget, metrics, and
  acceptance thresholds.
- [ ] Train matched multi-offset runs without changing the primary architecture or
  data split.
- [ ] Report next-token and per-offset losses separately.
- [ ] Rerun intrinsic, long-range, downstream, termination, and memory evaluations.
- [ ] Promote or reject the extension using the specification gates.

Exit gate: any accepted long-range gain is replicated and does not materially degrade
next-token loss, termination, stability, or runtime reliability.

## Phase 4: Termination and Replay Ablation

- [ ] Predeclare termination buckets, replay construction, decoding conditions, and
  matched prompt/sample seeds.
- [ ] Train the termination-head condition without replay.
- [ ] Train the generated-prefix replay condition if the head-only result is
  insufficient.
- [ ] Compare natural, syntax-constrained, replay-trained, and decoder-biased behavior
  without conflating them.
- [ ] Report length distributions, natural-stop, early-stop, and hard-cap rates plus
  primary loss and sequence-quality controls.
- [ ] Promote or reject the termination extension.

Exit gate: natural completion improves without forced-stop dependence, short-sequence
collapse, or material primary-quality regression.

## Phase 5: Biophysical Shape-Guidance Ablation

- [ ] Freeze and document the shape-encoder artifact, targets, training sources, and
  relationship to all heldout groups.
- [ ] Train the frozen-encoder condition from the corrected primary checkpoint.
- [ ] Train the jointly unfrozen condition with recorded discriminative learning
  rates and matched token exposure.
- [ ] Run grouped DNA-shape evaluation with one-hot, random-model, 5-mer, and 7-mer
  controls on shared folds.
- [ ] Rerun synonymous and de novo generation with matched seeds, confidence
  intervals, absolute effects, and memorization audits.
- [ ] Promote or reject shape guidance independently of termination and multi-offset
  objectives.

Exit gate: replicated shape-guided improvement survives sequence controls and does
not depend on leakage, unmatched decoding, or proxy-only interpretation.

## Phase 6: Combined Candidate and Final Report

- [ ] Combine only independently promoted extensions and predeclare the rationale.
- [ ] Train a matched combined candidate; do not replace independent ablations.
- [ ] Rerun the complete intrinsic, downstream, generation, memorization, runtime, and
  provenance suite.
- [ ] Publish a versioned comparison covering the basic model and every extension.
- [ ] Update repository headline tables while preserving a separate legacy section.
- [ ] Close or create follow-up issues based on failed gates and remaining limitations.

Exit gate: the selected model is supported by corrected replicated evidence, and each
claimed gain can be attributed to a tested extension.
