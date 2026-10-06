# AdamW vs SGD Optimizer Ablation for CodonLM

## Objective

Measure whether a simpler SGD optimizer is a viable alternative to the locked
AdamW optimizer for CodonLM's autoregressive next-codon training. Preserve the
current AdamW checkpoint and configuration as the production baseline unless a
predeclared, replicated comparison supports promotion.

## Motivation

The attached paper, *Do We Need Adam?*, studies SGD versus AdamW in online
reinforcement learning from verifiable rewards. That is a different optimization
regime from CodonLM's next-token likelihood training, so its result motivates an
ablation but does not predict its outcome here. SFT is a training stage/objective,
not an optimizer; this track compares optimizers while keeping the training task
fixed.

## Scope and Controls

- Compare the existing AdamW baseline with plain SGD under the same causal
  next-codon objective, frozen genome-held-out data, architecture, tokenization,
  initialization protocol, batch exposure, and token budget.
- Tune optimizer-specific peak learning rate using validation only; freeze the
  schedule shape, warmup fraction, weight-decay policy, and tuning grid before
  training. Report any optimizer-specific implementation difference explicitly.
- Do not use the test split for optimizer selection. Evaluate the selected runs on
  frozen test once, with the same unigram/bigram/trigram baselines.
- Keep SFT, RL, auxiliary losses, data changes, and architecture changes out of
  this experiment.

## Metrics

- Primary: unsmoothed held-out next-codon NLL and perplexity, with paired
  window-level uncertainty and per-seed results.
- Secondary: optimizer updates/second, wall time, peak memory, finite-update and
  resume behavior.
- Generation guardrails: natural stop, premature stop, hard-cap rate, and length
  distribution using fixed prompts/seeds and unrestricted decoding.

## Decision Rule

Do not promote SGD based on training loss, one seed, or the attached RL paper.
Require a replicated test-NLL result that improves or is non-inferior to AdamW
within a predeclared tolerance, does not regress generation/termination guardrails,
and provides a meaningful operational benefit (for example, reduced memory or
improved throughput). Otherwise retain AdamW and record the result as negative.

## Non-Goals

- Do not replace or invalidate the current two-seed AdamW baseline.
- Do not interpret low training perplexity alone as model quality or evidence of
  generalization.
- Do not equate SGD with supervised fine-tuning or RLVR.
