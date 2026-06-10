# Specification: Data Organization Consolidation

## Overview
Currently, the outputs of a run are scattered across three distinct locations in the workspace:
1. `outputs/checkpoints/<run_id>/` (weights and configs)
2. `outputs/scores/<run_id>/` (training history curves and metrics)
3. `runs/<run_id>/` (evaluation reports, charts, and tables)

This track consolidates all run-specific outputs into a single, unified directory under `runs/<run_id>/`. It ensures a clean, self-contained run layout that makes archiving, deleting, and sharing specific experiment runs simple.

## Objectives
1. **Unified Storage Layout**: Consolidate weights, scores, and diagnostics under a single run folder: `runs/<run_id>/checkpoints/` and `runs/<run_id>/scores/`.
2. **Git Hygiene Preservation**: Update the root `.gitignore` to ignore the heavy weights folder (`runs/**/checkpoints/`) while allowing the lightweight logs, CSVs, and charts to be checked in.
3. **Safety & Compatibility**: Write robust validation tests to verify compatibility and support fallback reading from legacy runs.
4. **Documentation**: Update the manual to reflect the consolidated folder structure.

## Technical Details
- **Ignored Directory Pattern**:
  ```gitignore
  runs/**/checkpoints/
  ```
- **Modified Scripts**:
  - `src/codonlm/train_codon_lm.py` (update checkpoints & scores output path)
  - `src/protein_lm/train_multi_task.py` (update checkpoints output path)
  - `src/eval/aggregator.py` and `scripts/compare_runs.py` (update scores reading path)
  - Diagnostic steps in `scripts/analyze_attention.py`, `scripts/analyze_embeddings.py`, and `scripts/sanity_kpis.py` (update paths)
