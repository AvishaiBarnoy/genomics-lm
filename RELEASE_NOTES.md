# Genomics‑LM Release Notes

This document tracks notable changes to the project. We follow a simple date‑based scheme and keep an "Unreleased" section for ongoing work. Tag commits as needed.

## Unreleased

- Analysis unification: `analysis.sh` now runs end‑to‑end analysis, including perplexity/KPIs, sequence quality, SS/disorder heuristics, calibration, and optional prefix plots.
- Reference tables: `scripts/build_reference_tables.py` builds per‑organism `codon_usage.tsv` and `cai_weights.tsv` to avoid duplicates (outputs under `data/reference/<name>/`).
- Sequence quality verifier: `scripts/seq_quality.py` computes ORF integrity, length/GC%, codon usage KL/JS vs reference, CAI (if provided), 3‑nt FFT periodicity, diversity/novelty (k‑mer Jaccard + MinHash). Merges KPIs into `metrics.json`.
- Calibration metrics: `scripts/calibration_metrics.py` computes ECE/Brier on a chosen split using a checkpoint.
- Secondary‑structure pipeline: `scripts/ss_propensity.py` (heuristic) and `scripts/probe_ss_linear.py` (supervised probe) with documentation.
- Disorder heuristics: `scripts/disorder_heuristics.py` adds CH‑plane, disorder fractions, low‑complexity segments; merges small KPIs into `metrics.json`.
- Long protein generation: `scripts/eval_generation_prefix.py` gains AA‑length controls; `src/codonlm/generate.py` adds constrained generation with terminal‑stop logic; new summary/plots.
- Length‑normalized GQS and combined prefix plots: `--gqs_normalize` option; `scripts/plot_eval_prefix.py` dual‑axis plot.
- Model toggles (off by default except tie): `tie_embeddings` (default true), optional GQA (`n_kv_head`), `use_sdpa`, alias for `grad_checkpointing`; example config `configs/m2_depth_upgrade.yaml`.
- Tokenizer/compat fixes: canonical specials (`<BOS_CDS>`, `<EOS_CDS>`, `<SEP>`) with legacy stoi aliases; consistent vocab size 68.

## 2025‑11‑05

- Multi‑dataset pipeline prepare with integrity checks and combined manifest.
- CDS‑only tokenizer and segment masking across `<SEP>` boundaries.
- Long generation benchmarking scaffold and compare‑runs scan mode.
- Stage‑2 classifier scaffold (embeddings + k‑mer baselines).

## 2025‑10‑27

- Initial training/evaluation pipeline, artifacts collection, and 6‑step interpretability scripts.

---

Guidelines for maintainers:
- Add entries to "Unreleased" during development (link PRs if desired). On tagging a release, copy Unreleased to a dated section and clear Unreleased.
- Keep entries concise and focused on user‑visible changes (new scripts/flags, behavior changes, deprecations).

