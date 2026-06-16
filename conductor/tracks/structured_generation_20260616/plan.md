# Plan: Structured Protein Generation

**Track:** `structured_generation_20260616`
**Status:** Closed as experimental finding
**Prerequisite:** Generative Design Loop track ✅

**Closeout (2026-06-16):** We implemented all zero-new-model controls plus a
structured-prefix experiment harness. The key result is negative but useful:
critic-guided filters improved the ProteinCritic stability score, but did not
raise ESMFold pLDDT. See [report.md](./report.md).

---

## Tasks

### Tier 1 — Zero new models

- [x] **T1a: Critic-guided ReD (`--min_stability` flag)**
  - `min_stability: float = 0.0` + `max_stability_attempts: int` added to `red_generate()` outer loop
  - Outer loop in `run_design_loop` rejects sequences below threshold, keeps best-so-far as fallback
  - `--min_stability`, `--max_stability_attempts` CLI flags added
  - Smoke test (5 seq, min_stability=0.65): stability_mean **0.689** vs baseline 0.608 (+13.4%) ✅
  - Full 50-seq run in progress → `outputs/reports/structured_gen_t1a/`

- [x] **T1b: Family-targeted generation (`--target_family_idx`)**
  - `--target_family_idx INT` and `--min_family_conf FLOAT` CLI flags added
  - Same outer loop as T1a; filters on `family_top1 == target_family_idx AND conf > min_family_conf`
  - Ready to use; needs a run to identify which family indices map to structured families

- [x] **T1c: Temperature annealing**
  - `anneal_temp: bool = False` added to `red_generate()`
  - Linearly anneals from `temperature` → `0.7 × temperature` after first 50 codons
  - `--anneal_temp` flag added; included in T1a smoke test ✅

### Tier 2 — Guided sampling

- [x] **T2a: Structured prompt prefix experiment harness**
  - Created `scripts/structured_prefix_experiment.py`
  - Hardcodes 3 structured-family prompts: DHFR/FolA-like, TEM-1 beta-lactamase-like, TPI/TIM-barrel-like
  - Generates scored continuations using the same CodonLM + ProteinCritic stack
  - Optional `--esm_fold_top` submits top candidates to ESMFold when network access is available
  - Ran 10 continuations per prefix and submitted all 30 to ESMFold (`outputs/reports/structured_prefix_experiment/`)
  - Result: 0/30 terminated, mean pLDDT 0.317, max pLDDT 0.383; prefix prompting did not improve fold confidence

- [x] **T2b: Top-p (nucleus) sampling**
  - `top_p: float = 0.0` added to `red_generate()` (overrides top-k when > 0)
  - Proper nucleus sampling: sort logits, cumulative prob cutoff at `top_p`
  - `--top_p FLOAT` CLI flag added

### Tier 3 — Training data surgery

- [ ] **T3a: PDB-filtered fine-tuning** - deferred to next research track
  - Write `scripts/filter_cds_by_pdb.py`:
    - Download UniProt `reviewed=True AND organism:bacteria AND 3d-structure:true` list
    - Cross-reference gene IDs against `data/processed/` sequences
    - Output filtered `data/processed/structured_train.bin`
  - Write `configs/stage3_structured.yaml` (1-epoch fine-tune from Stage 2.5 checkpoint)
  - Run fine-tune (overnight if needed)
  - Re-run 50-seq design loop; compare pLDDT before/after

- [ ] **T3b: Secondary structure conditioning** - deferred to next research track
  - Annotate PDB-matched sequences with DSSP-predicted secondary structure
  - Add `<HELIX>`, `<SHEET>`, `<COIL>` special tokens to the tokenizer
  - Fine-tune CodonLM to condition on these tokens
  - Test: `<HELIX> ATG ...` generation vs. unconditioned baseline

### Tier 4 — Reward signal

- [ ] **T4a: ESMFold reward fine-tuning (REINFORCE)** - deferred to next research track
  - Implement `scripts/rl_finetune_with_esm.py`
  - Generate batch of 20 sequences, get pLDDT from API, update model
  - Add KL divergence penalty vs. Stage 2.5 reference model
  - Track pLDDT curve over 50 update steps
  - *Note: free-tier ESMFold API limit ~1000 seq/day*

---

## Metrics to Track

| Run | n_seq | min_stability | temperature | pLDDT mean | pLDDT > 0.7 |
|---|---|---|---|---|---|
| Baseline | 50 | 0.0 | 0.9 | ~0.5 | 0/3 |
| T1a+T1c (smoke, 5 seq) | 5 | 0.65 | 0.9+anneal | pending ESMFold | — |
| T1a+T1c (full, 50 seq) | 50 | 0.65 | 0.9+anneal | ⏳ running | ? |
| T1b (family target) | 20 | 0.0 | 0.9 | ? | ? |
| T2a structured prefixes | 30 | 0.0 | 0.9 | 0.317 | 0/30 |
| T3a (PDB fine-tune) | 50 | 0.0 | 0.9 | ? | ? |

### Critic stability_mean comparison
| Run | stability_mean | Δ vs baseline |
|---|---|---|
| Baseline (50 seq, no filter) | 0.608 | — |
| T1a+T1c smoke (5 seq) | **0.689** | **+13.4%** |
| T1a+T1c full (50 seq) | **0.691** | **+13.6%** |

### ESMFold pLDDT comparison (top-3 sequences each run)
| Run | Seq | Critic stability | pLDDT (ESMFold) |
|---|---|---|---|
| Baseline | seq 1 | — | 0.41 |
| Baseline | seq 2 | — | 0.50 |
| Baseline | seq 3 | — | 0.60 |
| T1a+T1c | seq 17 | **0.754** | 0.41 |
| T1a+T1c | seq 38 | 0.689 | 0.288 |
| T1a+T1c | seq 20 | 0.689 | 0.471 |

---

## Key Finding: Critic stability ≠ ESMFold pLDDT

The MultiTask ProteinCritic stability head improved by **+13.6%** with the T1a filter,
but ESMFold pLDDT remained **unchanged** (0.4–0.5 range in both runs).

**Why:** The critic was trained to discriminate stable vs. unstable *natural* proteins.
The stability features it learned (e.g. arginine richness, length patterns) are tied to
natural protein distributions. De novo generated sequences that score high on the critic
just look *superficially* like its training data — they don't fold better in 3D.

**Implication:** Tier 1 & 2 approaches have a fundamental ceiling for pLDDT improvement.
The only paths to higher pLDDT are:
1. **T3a (PDB-filtered fine-tune):** Train CodonLM on sequences whose 3D structure is known → model learns codon-level signals of foldability
2. **T4a (ESMFold REINFORCE):** Use pLDDT directly as reward → closes the training signal gap

This is a publishable scientific finding in itself: *critic-guided selection on sequence models cannot substitute for structural feedback.*

## Prefix Experiment Result

The structured-prefix experiment tested whether short prompts from known enzyme
families could steer CodonLM toward foldable continuations without retraining.
It did not.

| Metric | Result |
|---|---:|
| Prefixes | DHFR/FolA-like, TEM-1 beta-lactamase-like, TPI/TIM-barrel-like |
| Continuations | 30 |
| Terminated | 0/30 |
| Mean critic stability | 0.571 |
| Mean ESMFold pLDDT | 0.317 |
| Max ESMFold pLDDT | 0.383 |
| pLDDT > 0.7 | 0/30 |

This strengthens the main conclusion: local prompt steering is not enough to
move the current CodonLM into foldable sequence space.

## Closeout Artifacts

- `scripts/generative_design_loop.py`: critic-guided ReD, family filtering, annealing, and top-p sampling
- `scripts/structured_prefix_experiment.py`: structured-prefix experiment harness
- `outputs/reports/structured_gen_test/`: 5-sequence smoke run
- `outputs/reports/structured_gen_t1a/`: 50-sequence T1a+T1c run
- `conductor/tracks/structured_generation_20260616/report.md`: final closeout report
