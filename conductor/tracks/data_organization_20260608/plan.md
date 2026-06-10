# Implementation Plan: Data Organization Consolidation

## Phase 1: Directory Restructuring & Git Rules
Set up the unified target directory and ignore rules.

- [ ] Task: Update the root `.gitignore`.
    - Add rule: `runs/**/checkpoints/`
    - Verify that no active git-tracked files match this pattern.
- [ ] Task: Adjust training script target directories.
    - Update default output directory variables in `src/codonlm/train_codon_lm.py` and `src/protein_lm/train_multi_task.py`.
    - Checkpoints path -> `runs/<run_id>/checkpoints/`
    - Scores/history path -> `runs/<run_id>/scores/`

## Phase 2: Diagnostic & Aggregator Updates
Update reading/writing paths in post-training scripts.

- [ ] Task: Update `src/eval/aggregator.py`.
    - Adjust paths to resolve `scores/` and `checkpoints/` from the consolidated run folder.
- [ ] Task: Update comparison and visualization scripts.
    - Modify `scripts/compare_runs.py` to search the new directory structure.
    - Check attention, embeddings, and saliency scripts to ensure they resolve inputs correctly.

## Phase 3: Safety Tests & Documentation
Write verification tests and record in user manuals.

- [ ] Task: Implement `tests/test_data_consolidation.py`.
    - Write tests that create a mock run, write dummy checkpoints and scores under the new layout, and assert they are parsed successfully by the aggregator and visualizer.
    - Test backwards-compatibility fallback logic (ensuring old legacy directories are still readable).
- [ ] Task: Document changes.
    - Update `MANUAL.md` and `GEMINI.md` to describe the new folder structure.
