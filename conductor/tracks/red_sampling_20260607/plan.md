# Plan: ReD Sampling for Genomic Inference

## Phase 1: Research & Strategy
- [x] Analyze `src/codonlm/generate.py` and `scripts/eval_generation_prefix.py`.
- [x] Verify current termination rates (found 0% in latest run).
- [x] Confirm applicability of ReD method from paper.

## Phase 2: Implementation - Core Generator
- [ ] Implement `generate_cds_red` in `src/codonlm/generate.py`.
    - Function signature: `generate_cds_red(model, device, ctx_ids, stoi, itos, target_codons, hard_cap, max_attempts=10, ...)`
    - Logic: Perform independent `generate_cds_constrained` calls, returning immediately on the first success (terminal stop).
- [ ] Implement `batch_red_sampler` to manage budget across multiple prefixes.

## Phase 3: Implementation - Benchmarking Script
- [ ] Create `scripts/benchmark_red.py`.
    - Support `--run_id`, `--max_genes`, `--global_budget` (in tokens).
    - Compare two modes: `standard` (sequential, fixed samples) vs `red` (round-robin, early discard).
    - Export `coverage_curve.csv` and `metrics.json`.

## Phase 4: Validation & Reporting
- [ ] Run benchmark on `2026-06-05_stage2.5_6L4H_d256_e10`.
- [ ] Generate comparison plots.
- [ ] Document findings in `DEVELOPMENT_LOG.md`.
