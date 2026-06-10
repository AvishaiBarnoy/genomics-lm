# Genomics‑LM Release Notes

This document tracks notable changes to the project. We follow a simple date‑based scheme and keep an "Unreleased" section for ongoing work. Tag commits as needed.

## Unreleased

- **SOTA Benchmarking & Compute Footprint Profiling**: Added a target prokaryotic evaluation suite comparing our models against published SOTA prokaryotic models (Evo 1 and GenSLM). Implemented zero-shot mutation scoring (`scripts/benchmark_zero_shot_mutations.py`) on protein and rRNA DMS, and linear probing (`scripts/benchmark_gene_essentiality.py`) on gene essentiality datasets. Added comparison reporting (`scripts/generate_sota_report.py`) demonstrating orders of magnitude higher compute-efficiency density ratio for locally-trained models on consumer hardware.
- **Stage 2.5: Genomic Tapes & Anchored Operons**: `src/codonlm/extract_genomic_tape.py` and `extract_anchored_operons.py` enable training on contiguous chromosomal data and targeted gene-boundary transitions.
- **Structural DNA Probing**: `scripts/probe_structural_awareness.py` validates model physics by mapping hidden states to Roll, EP, and MGW using DNAshape heuristics.
- **Stage 3: Multi-Task Protein Critic Implemented & Trained**: Completed training of the `MultiTaskProteinClassifier` (`configs/protein_critic.yaml` with an 8L8H_d256 architecture) over 50 epochs on Apple Silicon GPU (`mps`), achieving 76.81% validation accuracy on thermodynamic stability, 6.15% on Pfam (1,000 classes), and 5.50% on EC function (500 classes). Added full train-resumption features (`--resume`) and automatic epoch-end checkpoints (`last_critic.pt`).
- **Stage 3 Preparations**: `scripts/fetch_uniprot_metadata.py` and `scripts/build_multitask_dataset.py` introduced to create unified Pfam/Stability datasets for the Protein-Critic.
- **Hardware Optimization**: Re-enabled SDPA (Flash-like attention) in `model_tiny_gpt.py` providing a 3x speedup on Apple Silicon, alongside gradient accumulation fixes to respect 8GB RAM ceilings.
- **Dashboard Automation**: `scripts/web_dashboard.py` upgraded with retroactive Analysis execution, Structural Audit views, and Plain English biological summaries.
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

