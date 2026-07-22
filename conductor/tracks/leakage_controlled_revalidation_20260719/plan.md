# Leakage-Controlled CodonLM Revalidation Plan

## Status
In progress. Engineering gates, the local corrected dataset freeze, and evaluator
contracts are complete; full retraining remains blocked on the formal pipeline
freeze, runtime-policy selection, and immutable training configuration gate.

## Phase 1: Governance and CI
- [x] Relabel legacy results and unsupported claims (#87).
- [x] Make global grouped splitting the mandatory scientific default (#79).
- [x] Require core CPU tests, fatal lint, coverage artifacts, and clean-checkout
  verification in GitHub Actions (#91).
- [ ] Keep this plan and issue #92 synchronized when a gate changes status.

## Phase 2: Correct the Dataset Representation
- [x] Split CDS records at ambiguous codons without creating false adjacency; retain
  source IDs and oriented codon coordinates; report fragment statistics (#80).
- [x] Replace suffix truncation with lossless chunks and expose record, chunk,
  coordinate, boundary, and continuation metadata (#78).
- [x] Define and test a versioned dataset manifest schema covering sources, splits,
  fragments, chunks, packing, vocabulary, seeds, and artifact hashes.

Exit gate: fixtures prove that no transition crosses ambiguity, no source transition
is lost or duplicated, and every derived fragment/chunk retains its source split.

## Phase 3: Enforce Leakage and Training Correctness
- [x] Fail preparation on cross-split exact duplicates and disallowed protein
  homology; record offending IDs, thresholds, commands, and tool versions (#77).
- [x] Abort and clear an accumulation group after any non-finite loss, preserving
  correct optimizer/scheduler/resume counters (#83).
- [x] Apply configured attention dropout consistently in SDPA and manual paths (#81).
- [x] Resolve new-run vocabulary exclusively from the tokenizer artifact and fail on
  dataset, config, or checkpoint mismatch (#84).
- [x] Run CPU integration tests and MPS smoke train/save/resume tests on the corrected
  representation.

Exit gate: all blocking data and trainer PRs are merged, CI is green, and an MPS
preflight completes without OOM, non-finite updates, provenance gaps, or leakage.

## Phase 4: Freeze Corrected Datasets
- [x] Pin the source snapshot and create immutable genome-held-out and genus-held-out
  source-record manifests with split/packing seed `1337`.
- [x] Generate corrected fragment, chunk, packed, and vocabulary artifacts from the
  same source inventory.
- [x] Quarantine and verify zero exact cross-split CDS copies; complete mandatory
  protein-cluster and nearest-neighbor reports under the grouped-holdout policy.
- [x] Record manifest hashes, group/record/fragment/chunk/token counts, achieved
  split fractions, ambiguity statistics, and audit reports.
- [ ] Tag the dataset schema and artifact generation as the pipeline freeze.

Local freeze `718417694607bed760fcb2335db1f65c96ef69cdae1612853e8778eef5ba8406`
contains genome dataset `da3dfce28b7a46b8640d75c7cb417c867137a99e004ea359d85784ff0c269db9`
and genus dataset `10f41e818182704bbe4f95fbd81eb8696047762a32f84d167a4101675945ab95`.
The freeze is a local acceptance artifact until this pipeline PR merges and a
clean-checkout reproduction confirms the same identities.

Exit gate: a clean checkout can reproduce byte-identical manifests and all blocking
audits pass. Any later semantic change creates a new dataset version.

## Phase 5: Freeze Evaluation Instruments
- [x] Make perplexity baselines storage-format independent and vocabulary safe (#82).
- [x] Remove unsafe embedding fallbacks and record causal extraction provenance (#86).
- [x] Add gene/genome-grouped DNA-shape folds plus one-hot, random-model, 5-mer, and
  7-mer controls (#88).
- [x] Add protein-cluster-held-out AMR splits, class reporting, stratified bootstrap,
  and output isolation (#89).
- [x] Validate every evaluator on fixtures derived from the frozen manifest schema.

Exit gate: all evaluators consume explicit frozen artifacts, reject incompatible
inputs, preserve grouping, and emit machine-readable provenance.

## Phase 6: Train Corrected Models
- [ ] Select the MPS runtime policy through the equal-token quality gate in the
  optimized training validation track; record the selected policy without changing
  architecture or objectives.
- [ ] Train genome-held-out models from random initialization at seeds `1337` and
  `2027` with matched non-PAD token exposure.
- [ ] Run a separately labeled genus-held-out training protocol.
- [ ] Preserve configs, source/dataset hashes, checkpoints, logs, optimizer counters,
  consumed-token counters, wall time, peak memory, and failure metrics.

Exit gate: two genome-held-out seeds and the declared genus-held-out run finish with
passing manifests, no invalid updates, and complete checkpoint/resume provenance.

## Phase 7: Evaluate and Publish
- [ ] Produce the uniform/unigram/bigram/trigram/CodonLM intrinsic table with loss,
  perplexity, bits/codon, token count, and improvement over the best simple baseline.
- [ ] Extract corrected causal embeddings and rerun EC, essentiality, AMR, and
  DNA-shape evaluations under their controlled splits and shared controls.
- [ ] Run nucleotide/protein nearest-neighbor and training-match coverage audits for
  generated sequences before reporting memorization or novelty.
- [ ] Publish a versioned report with exact commands, hashes, per-seed and aggregate
  results, confidence intervals, limitations, and failed gates.
- [ ] Update README and benchmark documents from the versioned report while retaining
  a clearly separated legacy-versus-corrected table.

## Deferred Follow-Ups
- [x] Separate raw, CDS-constrained, and guided generation protocols (#85).
- [x] Implement batch-aware memmap conversion with fixed/dynamic parity tests (#90).
- [ ] Record whether the merged batch-aware path improves useful-token throughput on
  MPS without a memory regression before freezing the runtime policy.
