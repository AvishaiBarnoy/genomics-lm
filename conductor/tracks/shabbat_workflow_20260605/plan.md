# Implementation Plan: Shabbat 26-Hour Automated Workflow

## Phase 1: Preparation [checkpoint: ]
- [ ] Task: Create `configs/stage2.5_extended.yaml`. This configuration will resume the current 6L Master model and fine-tune it for 10 additional epochs with a lower learning rate to encourage proper termination.
- [ ] Task: Create `shabbat_workflow.sh`. This bash script will:
  1. Wait for the active training process (PID 90542) to complete.
  2. Automatically run the evaluation pipeline (Inference Benchmark, Motif Audit).
  3. Launch the `stage2.5_extended.yaml` training run.
  4. Launch the `m2_max_10L8H.yaml` contingency training run.

## Phase 2: Execution [checkpoint: ]
- [ ] Task: Make `shabbat_workflow.sh` executable.
- [ ] Task: Launch `shabbat_workflow.sh` as a background process so it runs autonomously over the 26-hour period.
