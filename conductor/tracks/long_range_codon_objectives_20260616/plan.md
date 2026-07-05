# Long-Range CodonLM Objectives Plan

## Status

In progress. The first implementation pass adds config-gated multi-offset
future-token losses, next-token-preserving metrics, and whole-gene pack audits.

## Tasks

- [x] Add a config-gated multi-offset LM loss.
- [x] Add tests for offset target creation and end-of-sequence masking.
- [x] Keep next-token perplexity separate from auxiliary-loss validation.
- [x] Add whole-gene/truncation audit metadata for dynamic packs.
- [x] Add a smoke config with offsets `+4`, `+8`, `+16`, `+32`.
- [x] Run a wall-time-bounded MPS smoke experiment with the long-range objective enabled.
- [x] Add periodic checkpointing and per-run logging before rerunning long MPS jobs.
- [x] Compare validation next-token perplexity against standard training.
- [x] Run matched quick biological prefix evaluation for Stage 2.6, Stage 3, and long-range.
- [x] Add a config-gated termination distance auxiliary objective.
- [x] Add decoder-side stop-bias support from the termination auxiliary head.
- [x] Rescore generated libraries with calibrated ProteinCritic selection rules.
- [x] Add generated-prefix replay or denoising corruption.
- [x] Build an offline hard-negative dataset builder from generated ORFs.
- [x] Constrain hybrid-vocab CDS generation to codon tokens by default.
- [x] Build quick replay JSONL from the physical-termination checkpoint.
- [x] Smoke-test replay fine-tuning and wall-time checkpointing on MPS.
- [x] Run generated-prefix replay fine-tuning from the physical-termination checkpoint.
- [x] Re-evaluate replay fine-tune with matched prefix generation and ProteinCritic scoring.
- [x] Run a conservative d384-vs-d512 capacity ablation only after objective/data metrics improve. (Resolved: bypassed via parameter-efficient backbone freezing).

## Biology-Informed Priors

- Helical local structure often creates useful residue relationships around
  nearby offsets such as `i -> i+3/i+4`.
- Medium-range sequence constraints matter for motifs, beta-strand pairing,
  domain cores, and topology, but they are weakly supervised by pure next-token
  loss.
- The multi-offset objective is therefore a diagnostic training pressure, not a
  complete structural objective.

## Initial Experiment

Use `configs/long_range_offsets_smoke.yaml` from a backed-up Stage 2.6 checkpoint.
Accept the objective only if:

- next-token validation perplexity regresses by no more than 2%;
- generation does not collapse into short peptides or non-terminating outputs;
- calibrated ProteinCritic top-fraction enrichment improves for at least one
  structural/useful label without broad degradation.

## Smoke Result

- Run id: `2026-06-17_long_range_offsets_smoke_mps_b4`
- Device/config: MPS, `batch_size=4`, offsets `+4/+8/+16/+32`, AMP disabled.
- Result: the objective trained stably and saved `last.pt` at the 60-minute
  wall-time limit, reaching optimizer step 30.
- Limitation: the run did not complete an epoch or validation, so no
  `val_next_loss`/perplexity comparison is available yet.
- Finding: MPS AMP fails on the offset-loss backward path; full precision is
  stable but too slow for full validation under the current smoke config.

## Long-Run Failure Audit

- Run id: `2026-06-17_long_range_offsets_mps_b4_6h5`
- Status: ended non-cleanly before epoch end or wall-time handler saved a checkpoint.
- Available files: `checkpoints/config.yaml` and `scores/curves.csv` header only.
- Missing evidence: no `last.pt`, `best.pt`, `meta.json`, or run log existed, so the
  cause cannot be distinguished between MPS/runtime error, process kill, memory
  pressure, shell/session interruption, or sleep/energy handling.
- Mitigation implemented: shared run logging writes `runs/<RUN_ID>/logs/train.log`,
  and `checkpoint_every_steps` / `checkpoint_every_minutes` can save `last.pt`
  periodically during long epochs.

## Matched Quick Biological Evaluation

- Runs compared with `scripts.eval_generation_prefix --preset quick --seed 1337`.
- Stage 2.6, Stage 3, and long-range all had `terminal_stop_rate=0.0` and
  `hard_cap_rate=1.0`.
- Mean AA identity stayed near `0.077` across all three runs.
- Long-range validation loss improved slightly, but biological generation was
  effectively flat versus Stage 2.6.

## Termination Auxiliary Follow-Up

Implemented an optional supervised distance-to-stop objective:

- Config: `termination_loss_enabled: true`.
- Labels: derived on the fly from the shifted target tensor; no NPZ format change.
- Buckets: immediate stop, very near, near, far, and no/very-far stop.
- Smoke config: `configs/termination_aux_smoke.yaml`.

Acceptance criterion: a termination-aux run must improve terminal-stop behavior
or reduce hard-cap rate under matched prefix evaluation without degrading
next-token perplexity beyond the accepted tolerance.

### Training Result

- Run id: `2026-06-18_termination_aux_mps_b4_v1`
- Device/config: MPS, `batch_size=4`, 2 epochs.
- Validation: `val_next_loss=4.0868` (`ppl=59.55`), close to Stage 2.6.
- Auxiliary learning: `train_term_loss=0.8993`, `val_term_loss=0.7934`.
- Matched quick prefix eval without decoder use: `terminal_stop_rate=0.0`,
  `hard_cap_rate=1.0`, mean AA identity `0.0756`.

Interpretation: the auxiliary head learned a distance-to-stop representation, but
training alone did not route that signal back into the token logits at inference.
The head has to be consumed by the decoder/sampler.

### Decoder-Side Stop Bias

Implemented optional generation-time stop-codon biasing:

- `src.codonlm.generate.generate_cds_constrained(..., termination_bias_enabled=True)`
- CLI flags on `scripts.eval_generation_prefix`:
  - `--termination_bias`
  - `--termination_stop_bias`
  - `--termination_trigger_class_max`
  - `--termination_bias_window`

Matched quick eval:

- Run id: `2026-06-18_termination_aux_mps_b4_v1`
- Output label: `gen_prefix_quick_seed1337_stopbias5_window5_anyclass`
- Command shape: `--preset quick --seed 1337 --termination_bias
  --termination_stop_bias 5.0 --termination_trigger_class_max 4
  --termination_bias_window 5`
- Result: `terminal_stop_rate=1.0`, `hard_cap_rate=0.0`,
  `early_stop_rate=0.0`.
- Length: mean `100.86` codons, median `101`, range `97-107` around
  `target_codons=100`.
- Quality: mean AA identity `0.0756`, effectively unchanged from the no-bias
  eval; median GQS improved from roughly `26-27` to `46-57` because the generated
  sequences now receive valid stop/frame credit.
- Bias behavior: 72/80 samples needed one biased step, 7/80 needed two, and
  1/80 needed three. The head predicted class `4` for all samples near the
  target-length boundary, so strict immediate-stop triggering did not fire.

Conclusion: the decoder fix solves the valid-gene-ending failure without the
short-peptide collapse caused by ungated global stop bias. It does not, by
itself, improve protein semantic similarity or structural function.

## Physical Termination Transfer Follow-Up

- Run id: `2026-06-19_physical_termination_transfer_mps_b4_e1`
- Objective: transfer from the termination-aux checkpoint into a hybrid CDS+UTR
  setting that exposes downstream nucleotide context around CDS ends.
- Dataset: 91,131 hybrid CDS+UTR examples from 24 local GBFF files; packed into
  70,650 train windows, 8,433 validation windows, and 8,491 test windows.
- Training result: completed 3 epochs. Validation improved every epoch:
  `val_loss` 5.496 -> 5.072 -> 4.882, with epoch 3 saved as `best.pt`.
- Matched quick prefix evaluation:
  - Stage 2.6 baseline: terminal stop 0%, hard-cap 100%, median GQS 26.62,
    mean AA identity 0.0769.
  - Termination-aux: terminal stop 0%, hard-cap 100%, median GQS 26.44,
    mean AA identity 0.0756.
  - Physical transfer: terminal stop 0%, hard-cap 100%, median GQS 21.40,
    mean AA identity 0.0947.
- Nonzero termination-bias evaluation with `--termination_stop_bias 8` did not
  change behavior. The auxiliary head predicted class 4 ("far/no stop") for all
  generated samples, so strict stop-bias decoding never activated.
- Report:
  `runs/2026-06-19_physical_termination_transfer_mps_b4_e1/scores/physical_termination_eval_report.md`

Interpretation: hybrid CDS+UTR transfer improves local AA-prefix similarity but
does not solve natural gene termination. More epochs on the same teacher-forced
objective are unlikely to fix the observed generation failure, because the
termination head is wrong specifically on generated off-distribution states.

## Generated-Prefix Replay Implementation

Implemented the first hard-negative replay path:

- `scripts/build_generated_prefix_replay.py` generates prefix continuations from
  an existing checkpoint, keeps hard-cap failures without terminal stops, and
  writes JSONL records with sparse termination-class labels around the target
  boundary.
- Hybrid-vocab CDS generation is now codon-constrained by default. The smoke
  build exposed that the physical-transfer model could emit single-nucleotide
  UTR tokens during CDS continuation; this made "100 codons" expand to hundreds
  of tokens. `src.codonlm.generate.generate_cds_constrained` now masks generated
  CDS continuations to codon tokens unless `--allow_non_cds_tokens` is passed by
  an evaluation/replay diagnostic command.
- Prefix contexts now omit `<EOS_CDS>`. `scripts.query_model.dna_prefix_to_ids`
  is used for generation prompts, while `dna_to_ids` remains available for full
  CDS scoring contexts.
- `src.codonlm.replay.GeneratedTerminationReplayDataset` loads those records,
  left-clips long generated contexts to `block_size`, and preserves absolute
  label positions.
- `src.codonlm.train_codon_lm` now supports:
  - `replay_loss_enabled`
  - `replay_data`
  - `replay_loss_weight`
  - `replay_batch_size`
- `configs/physical_termination_replay.yaml` transfers from
  `2026-06-19_physical_termination_transfer_mps_b4_e1/checkpoints/best.pt` and
  adds replay loss on top of the normal next-token and termination-distance
  objectives.

Replay labels are intentionally sparse and auxiliary-head only. This is a
conservative correction for generated off-distribution states; it should be
evaluated before adding direct stop-token loss or larger model capacity.

Implemented/run so far:

```bash
python -m scripts.build_generated_prefix_replay \
  --run_id 2026-06-19_physical_termination_transfer_mps_b4_e1 \
  --ckpt best.pt \
  --device mps \
  --preset quick \
  --out runs/2026-06-19_physical_termination_transfer_mps_b4_e1/scores/generated_prefix_replay.jsonl
```

Result: 80/80 quick-prefix generations were hard-cap failures and became replay
records. Corrected record lengths are 102-111 tokens, with no `<EOS_CDS>` in the
prefix context and 13 sparse labels per record.

Replay training smoke:

```bash
caffeinate -i python -m src.codonlm.train_codon_lm \
  --config configs/physical_termination_replay.yaml \
  --run_id 2026-06-24_physical_termination_replay_smoke \
  --max_time_minutes 3
```

Result: MPS smoke loaded 80 replay records, transferred from the physical
checkpoint, completed 2 optimizer steps, and saved `last.pt` through the
wall-time handler at microbatch 64.

Next run:

```bash
caffeinate -i python -m src.codonlm.train_codon_lm \
  --config configs/physical_termination_replay.yaml \
  --run_id 2026-06-24_physical_termination_replay_mps_b4_e1
```

Acceptance criterion: replay fine-tuning should reduce hard-cap rate under
matched quick prefix evaluation without erasing the physical-transfer AA-identity
gain or substantially worsening next-token validation.


## Separate-Heads Multi-Offset Architecture & Evaluations (Helical Run v1)

- **Architecture Design**: Added config-gated separate projection heads (`offset_projs`) for each target $x \in \{4, 8, 16, 32\}$ in `TinyGPT`. This isolates the future-prediction auxiliary targets from the primary next-token prediction task, preserving causal next-token perplexity (unlike the legacy shared-head attempt which caused target smearing).
- **Run ID**: `separate_heads_full_run`
- **Config**: `configs/separate_heads_full.yaml`
- **Training Results**:
  - Converged over 2 epochs.
  - Standard validation next-token perplexity was completely preserved at **`59.478`** (matching the Stage 2.6 baseline of `59.55`).
  - Validation offset losses trained stably (`val_offset_4 = 4.0931`).
- **ProteinCritic Integration**: Updated `eval_generation_prefix.py` to support `--critic_stability` scoring, using `best_critic.pt` and `protein_critic.yaml` to measure thermodynamic stability and Pfam/EC classification confidence during generation.
- **Evaluation Results**:
  - Matched quick evaluations on the 2-epoch checkpoint showed a systematic increase in thermodynamic stability probability (`mean_critic_stability` increased across all conditions, with a **+1.2% absolute improvement** at $k=3$ from `0.5686` to `0.5757`).
  - Perplexity/generation drift stability (`ppl_stability`) improved from **0.898 to 0.917**, proving the look-ahead priors anchor sequences against autoregressive drift.

## Beta-Sheet Strand Run Setup (v2: x=[2, 8, 16, 32])

- **Objective**: Introduce the $x=2$ offset target corresponding to alternating $\beta$-strand side-chain orientations (which face the same plane and pack together, stabilizing sheets).
- **Config**: `configs/separate_heads_v2.yaml`
- **Weight Transfer**: Initialized from `separate_heads_full_run` best checkpoint. It successfully transferred all 182 core parameter matrices, discarded the unused $x=4$ head, and initialized the new $x=2$ head to identity mapping.
- **Status**: Launched as background task `separate_heads_v2`.
