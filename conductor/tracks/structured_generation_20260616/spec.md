# Track: Structured Protein Generation

**Created:** 2026-06-16
**Depends on:** Generative Design Loop track (complete)
**Goal:** Improve the generated sequence library so that ESMFold pLDDT scores rise from
the current 0.4–0.6 range into the "confident structure" range (> 0.7), without
retraining from scratch or requiring GPU infrastructure.

---

## Problem Statement

The Generative Design Loop (Stage 12) produced 50 diverse sequences (pairwise identity 9.2%)
but all score pLDDT < 0.7 under ESMFold — classified as likely intrinsically disordered.

**Two separable root causes:**
1. **No selection pressure during generation** — ReD accepts any terminated ≥50 AA sequence regardless of predicted structure.
2. **Training data bias** — CodonLM was trained on all bacterial CDS including disordered regions, linkers, and frameshifts. The model has no preference for foldable sequence space.

---

## Improvement Tiers

### Tier 1 — Zero new models, implement today

**Task T1a: Critic-guided ReD (`--min_stability` flag)**
- Extend `red_generate()` to score each candidate with the MultiTask critic before accepting
- Reject sequences where `stability_prob < threshold` (suggested: 0.65)
- Expected effect: force model to search for sequences in the stable region of its distribution
- Cost: critic already loaded in the loop; ~5 ms overhead per candidate
- Implementation: add `min_stability: float = 0.0` parameter to `red_generate` and `run_design_loop`

**Task T1b: Family-targeted generation**
- Add `--target_family_idx INT` CLI flag
- ReD only accepts sequences where `family_top1 == target_family_idx AND family_top1_conf > 0.3`
- Target well-structured Pfam families (e.g. TIM barrels, AAA ATPases, globins)
- Provides a directed exploration mode vs. the current undirected library

**Task T1c: Temperature annealing**
- Anneal temperature from T=1.0 → T=0.7 as sequence grows past 50 codons
- Mimics how structured protein cores are more conserved than termini
- Trivial 2-line change in `red_generate`

### Tier 2 — Guided sampling, no retraining

**Task T2a: Structured prompt prefix**
- Feed the first 10–20 codons of a known well-structured bacterial protein as the prompt
- Targets: DHFR (Pfam PF00186), TEM-1 β-lactamase (PF00144), Triosephosphate isomerase (PF00121)
- The model continues from this biased context — tends to stay in the same fold class
- Test with `--prompt "ATG GTT ATT ACA GCA ..."` in the existing CLI

**Task T2b: Top-p (nucleus) sampling**
- Replace top-k sampling with top-p (p=0.9)
- Top-p is better calibrated for protein generation — avoids the hard cutoff of top-k
- Add `--top_p FLOAT` flag alongside existing `--top_k`

### Tier 3 — Training data surgery (1–2 hrs, ~1 epoch)

**Task T3a: PDB-filtered fine-tuning**
- Filter the existing 15-genome CDS dataset to sequences whose proteins have PDB structures
- Source: UniProt `reviewed=True AND organism:bacteria AND 3d-structure:true` → cross-reference Gene IDs
- Expected: ~15–30% of CDS survive the filter → retrain CodonLM for 1 epoch on this subset
- This directly teaches the model "what structured bacterial proteins look like at the codon level"
- Implementation:
  1. `scripts/filter_cds_by_pdb.py` — downloads UniProt structured-bacteria list, filters `data/processed/train.bin`
  2. Run 1-epoch fine-tune with `configs/stage3_structured.yaml`
  3. Re-run the design loop to compare pLDDT before/after

**Task T3b: Secondary structure conditioning**
- Annotate training sequences using DSSP predictions from paired PDB structures
- Prepend `<HELIX>`, `<SHEET>`, `<COIL>` tokens to codon sequences
- Fine-tune CodonLM to follow structural conditioning — generation becomes steerable
- This makes the generation "conditional": `<HELIX> ATG ...` → helix-biased continuation

### Tier 4 — Reward signal (1–3 days)

**Task T4a: ESMFold reward fine-tuning (REINFORCE)**
- Generate a batch of 20 sequences per step
- Submit to ESMFold API (free tier: ~1000/day)
- Use mean pLDDT as scalar reward → policy gradient update on CodonLM
- Conservative implementation: keep KL penalty vs. original model to avoid mode collapse
- Estimated API cost: ~500 sequences for a meaningful signal = 1 day of free-tier budget

---

## Success Criteria

| Milestone | Target |
|---|---|
| Critic-guided ReD (T1a) | ≥ 20% of accepted sequences have `stability_prob > 0.7` |
| PDB-filtered fine-tune (T3a) | pLDDT mean rises from 0.5 → 0.65 on 10-sequence sample |
| Full structured library | At least 5/50 sequences with pLDDT > 0.7 under ESMFold |
| Conditional generation (T3b) | Helix-conditioned seqs have measurably higher helix content vs. unconditioned |

---

## Recommended Implementation Order

1. **T1a** (critic-guided ReD) — 20 min, highest return per effort
2. **T1b** (family-targeted) — 30 min, enables directed conference demo
3. **T3a** (PDB-filtered fine-tune) — most principled fix; schedule as overnight run
4. **T2a** (structured prompt prefix) — quick experiment while T3a trains
