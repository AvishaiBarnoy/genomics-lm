# AdamW vs SGD Optimizer Ablation Plan

## Status

Proposed; training has not started. Wait until the genus-held-out run and the
corrected ProteinCritic/XGBoost comparison have cleared their current readiness
gates, unless the project explicitly reprioritizes this study.

## Plan

- [x] Record the distinction between training objective/stage (next-token
  pretraining, SFT, RLVR) and optimizer (AdamW, SGD).
- [x] Record the attached RLVR paper as motivation only, not direct evidence for
  next-codon pretraining.
- [ ] Freeze the paired run configs, data/checkpoint provenance, SGD learning-rate
  search, schedule, weight decay, equal-token budget, seed policy, and acceptance
  tolerances before launching.
- [ ] Run a bounded training pilot for SGD to validate finite gradients, scheduler
  behavior, checkpoint/resume, memory, and throughput.
- [ ] Train paired AdamW and SGD conditions from fresh initialization on the same
  stream and token budget; first screen on validation only.
- [ ] Replicate selected paired conditions across the predeclared seeds.
- [ ] Evaluate the frozen test once against the identical simple baselines; report
  NLL/PPL and paired uncertainty.
- [ ] Run fixed-prompt unrestricted-generation termination and length guardrails.
- [ ] Promote, retain as an experimental optimizer, or reject SGD using the
  predeclared decision rule; keep the AdamW baseline intact.

## Exit Gate

An independently reviewable report contains configs and hashes, per-seed outcomes,
paired uncertainty, operational cost, generation guardrails, and an explicit
decision. No optimizer is promoted from training loss or unreplicated evidence.
