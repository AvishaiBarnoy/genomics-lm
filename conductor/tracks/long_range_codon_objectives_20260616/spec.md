# Long-Range CodonLM Objectives

## Goal

Teach CodonLM longer-range coding constraints without replacing the standard
causal next-token objective. The first implementation adds future-token
auxiliary losses and whole-gene truncation audits so we can test whether the
current `d384` model is objective-limited before scaling to `d512`.

## Scope

- Add config-gated multi-offset losses for `n+4`, `n+8`, `n+16`, and `n+32`.
- Keep next-token cross entropy as the primary optimization and perplexity metric.
- Mask padding and future targets that cross `<EOS_CDS>` or `<SEP>` boundaries.
- Report offset losses separately from next-token perplexity.
- Audit dynamic packs for clipped-at-`block_size` long genes.

## Non-Goals

- Do not replace causal LM training.
- Do not start preference/RL training in the first pass.
- Do not approve `d512` scaling until objective/data ablations improve generated
  protein metrics without damaging termination or next-token perplexity.
