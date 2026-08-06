# Genomics-LM Workflow Guide

This guide establishes the mandatory scientific and engineering workflow for developers and AI agents working on the Genomics-LM codebase. It ensures all training and evaluation runs remain leakage-free, computationally efficient, and statistically validated.

---

## 1. Data Preparation and Leakage Prevention

### The Golden Rule
**Never stack independent sequence-level dataset splits.** This leads to P0 genomic and strain-level leakage.

### Mandatory Workflow
1. Define your dataset sources in a YAML config (e.g. `configs/tiny_mps.yaml`).
2. Run [`build_global_manifest.py`](../scripts/build_global_manifest.py) to extract CDS records, resolve stable accessions, group records globally by genome or genus, and partition groups before packing. This is also the default preparation route used by `main.sh`:
   ```bash
   python -m scripts.build_global_manifest --config configs/tiny_mps.yaml --run-id <RUN_ID> --run-dir runs/<RUN_ID> --group-by genome
   ```

   Scientific preparation requires MMseqs2 on `PATH`. Before tokenization or
   packing, the builder hashes normalized full CDS records, clusters translated
   proteins, and searches validation/test records against training at both the
   nucleotide and protein levels. It writes commands, the MMseqs2 version,
   thresholds, identity summaries, and offending source IDs to
   `leakage_audit.json`. The default policy makes cross-split exact CDS duplicates
   and protein clusters fatal.

   The default protein-cluster gate uses 30% sequence identity and 80% coverage.
   Override those recorded thresholds with `max_cross_split_protein_identity` and
   `min_homology_coverage` in the run config. `--skip-homology-audit` and
   `--allow-cross-split-exact-duplicates` exist only for fixtures and legacy
   reproduction; either marks the resulting manifest as non-scientific.

   Whole-genome and whole-genus holdouts may instead declare
   `exact_duplicate_policy: quarantine` and `protein_homology_policy: report`.
   Quarantine deterministically retains each exact-CDS family in test, then
   validation, then training priority and records every removed source. Homologous
   but non-identical genes remain in grouped splits because conserved protein
   families are part of the intended generalization problem; their cluster and
   nearest-neighbor distributions remain mandatory report artifacts. Dedicated
   protein-family or homology-held-out evaluations must retain the strict `block`
   policy.

   Scientific preparation also fails when fewer than three groups are available.
   `--allow-sequence-split` is an explicit development-only escape hatch and marks
   the resulting manifest as non-scientific. The older
   `scripts.pipeline_prepare` route is disabled unless
   `--allow-legacy-per-dataset-split` is supplied for historical reproduction.
3. Verify that the output splits contain mutually exclusive genome sets by inspecting the generated `data/processed/global/<RUN_ID>/cds_meta.tsv`.

4. Validate the content-addressed dataset contract before any scientific run:
   ```bash
   python -m scripts.validate_dataset_manifest \
     data/processed/global/<RUN_ID>/manifest.json
   ```
   See [`DATASET_MANIFEST.md`](DATASET_MANIFEST.md) for schema and compatibility
   rules. A missing manifest is legacy/unverified; a present but invalid manifest
   is fatal.

Generated CDS claims require a separate novelty report against the frozen training
source records:

```bash
python -m scripts.audit_generated_sequences \
  --train-fasta data/frozen/train_cds.fasta \
  --generated-fasta runs/<RUN_ID>/generated.fasta \
  --output runs/<RUN_ID>/scores/generated_leakage_audit.json
```

This records the nearest nucleotide and protein training neighbor for every
generated sequence and reports position coverage by exact 30-nt and 10-aa
training substrings.

---

## 2. Next-Codon Perplexity Baseline Evaluation

### The Golden Rule
**Never report perplexity (PPL) points in isolation.** Raw perplexity is meaningless without composition-matched baselines.

### Mandatory Workflow
For every test evaluation, calculate and compare model performance against Uniform and Markov baselines:
1. Run [`eval_ppl_baselines.py`](file:///Users/User/github/genomics-lm/scripts/eval_ppl_baselines.py) using the train and test splits:
   ```bash
   python -m scripts.eval_ppl_baselines \
     --train data/processed/global/<RUN_ID>/train_bs256.npz \
     --test data/processed/global/<RUN_ID>/test_bs256.npz \
     --manifest data/processed/global/<RUN_ID>/manifest.json \
     --config configs/tiny_mps.yaml \
     --output-prefix runs/<RUN_ID>/scores/genome_holdout_ppl_baselines
   ```
   The evaluator resolves the dataset-adjacent `itos.txt`, validates every token
   ID, and writes both JSON provenance and a Markdown table. Repeat with the
   genus-held-out dataset and a distinct output prefix.
2. Report the **excess bits per codon** ($\Delta H$) and the perplexity drop over the 2nd-order Markov (Trigram) baseline.

The baseline evaluator resets trigram history after `<SEP>` so its accessible
context matches the model's segment mask. If CodonLM does not beat bigram and
trigram, run the context diagnostic in
`docs/CONTEXT_LEARNING_DIAGNOSTICS.md` before downstream evaluation or extension
training.

Evaluate the corrected model on the same manifest-bound test artifact:

```bash
python -m scripts.evaluate_test \
  --run_dir runs/<RUN_ID> \
  --manifest data/processed/corrected/<FREEZE_ID>/genome/manifest.json
```

Corrected checkpoints fail if the explicit manifest is missing or if its dataset or
vocabulary identity differs from the checkpoint. `test_nll` and `test_ppl` are
computed from ordinary unsmoothed cross-entropy and are directly comparable with the
simple baselines. `test_objective_loss` separately records cross-entropy with the
checkpoint's configured label smoothing.

Evaluate a distinct final checkpoint without overwriting the selected-best result:

```bash
python -m scripts.evaluate_test \
  --run_dir runs/<RUN_ID> \
  --manifest data/processed/corrected/<FREEZE_ID>/genome/manifest.json \
  --checkpoint-name last.pt \
  --metric-prefix last_test
```

---

## Corrected ProteinCritic Dataset

Training runs are collision-safe. A fresh launch whose run ID already exists is
written to the next `-rNNN` directory. In-place continuation requires `--resume`
with that run's newest `last` checkpoint and a configured total epoch target greater
than its completed epoch count. Use a new run ID to fork from an older or best
checkpoint; best checkpoints are evaluation artifacts, not in-place resume points.

This lifecycle also applies to ProteinLM, the single-task protein classifier,
Protein EBM, and NoProp. ProteinLM/classifier checkpoints can resume at recorded
optimizer-safe microbatch boundaries using a deterministic epoch sampler. EBM and
NoProp currently resume only from completed epochs. NoProp checkpoints include the
embedding, per-block, and output-head optimizer states.

Build the homology-cluster-held-out critic artifacts before critic training:

```bash
caffeinate -i python -m scripts.build_corrected_protein_critic_dataset \
  --protein-records data/processed/protein_pfam_labels.json \
  --annotation-metadata data/processed/uniprot_metadata_full.csv \
  --stability-csv data/raw/stability/dG_extdG_data_Fig1.csv \
  --out-dir data/processed/protein_lm/corrected-v2 \
  --threads 2
```

The builder requires MMseqs2, clusters all sources together, assigns whole clusters
to one split, reserves stability clusters in every split, filters Pfam/EC labels by
post-split support, and writes `manifest.json` with input, tool, threshold, split,
and artifact hashes. Records without any retained Pfam, EC, or stability target are
discarded after the support gates. Train the corrected critic with:

```bash
caffeinate -i python -m src.protein_lm.train_multi_task \
  --config configs/corrected_protein_critic_v1.yaml
```

The trainer verifies every dataset artifact against the manifest before training,
treats `stability_score` as continuous regression, and stores the dataset provenance
and model specification in each checkpoint.

Before changing physical batch size or context on MPS, run the isolated critic
benchmark:

```bash
caffeinate -i python -m scripts.benchmark_protein_critic_training \
  --config configs/corrected_protein_critic_v1.yaml \
  --matrix configs/corrected_protein_critic_mps_benchmark.yaml \
  --out runs/corrected-protein-critic-mps-benchmark \
  --force-gpu
```

The corrected M2/8 GB selection is batch 2, accumulation 16, and context 512.
Context 256 is faster but truncates most Pfam/EC-labelled proteins and is not the
primary scientific configuration.

After the unweighted baseline is frozen, run the validation-selected class-balance
ablation with:

```bash
caffeinate -i python -m src.protein_lm.train_multi_task \
  --config configs/corrected_protein_critic_class_balanced_v1.yaml
```

This config changes only the Pfam/EC training objective. It computes square-root
inverse-frequency weights from training labels, leaves validation loss unweighted,
and keeps the test split sealed until the conductor promotion decision is recorded.

---

## 3. Synonymous and Shuffling Controls

### The Golden Rule
**Verify if the model learns codon-level biophysics or just amino acid maps.**

### Mandatory Workflow
1. Generate control datasets from your test split:
   ```bash
   python -m scripts.generate_synonymous_controls \
     --test_npz data/processed/corrected/<FREEZE_ID>/genome/test_bs256.npz \
     --manifest data/processed/corrected/<FREEZE_ID>/genome/manifest.json \
     --out_dir runs/<RUN_ID>/controls
   ```
   This outputs:
   - `test_control_synonymous_bs256.npz` (random synonymous codons, same protein).
   - `test_control_codon_shuffle_bs256.npz` (codons shuffled within each CDS, same codon composition).
   - `test_control_protein_shuffle_bs256.npz` (amino acids shuffled within each CDS, same amino-acid composition).
2. Run `evaluate_test.py` on each control with its generated provenance sidecar:
   ```bash
   python -m scripts.evaluate_test \
     --run_dir runs/<RUN_ID> \
     --test_npz runs/<RUN_ID>/controls/test_control_synonymous_bs256.npz \
     --derived_provenance runs/<RUN_ID>/controls/test_control_synonymous_bs256.npz.provenance.json \
     --manifest data/processed/corrected/<FREEZE_ID>/genome/manifest.json
   ```
   Corrected evaluation fails if the control, source test artifact, vocabulary, or
   dataset identity differs from the recorded derivation.
3. Assert that the model's perplexity is significantly better (lower) on the natural test set compared to the synonymous recoding set.

---

## 4. Downstream Probe Controls

### The Golden Rule
**Always control for token-identity decodability in shape and classification probes.**

### Mandatory Workflow
Regenerate embeddings from the corrected checkpoint before running any probe:

```bash
python -m scripts.extract_embeddings \
  --run_dir runs/<RUN_ID> \
  --fasta data/frozen/test_cds.fasta \
  --manifest data/processed/corrected/<FREEZE_ID>/genome/manifest.json \
  --out runs/<RUN_ID>/embeddings/test_causal.npz
```

Extraction requires the canonical causal `forward_hidden()` model API and the
run-resolved vocabulary. Shape-guided checkpoints must contain their trained
shape encoder. The adjacent `.npz.metadata.json` sidecar records checkpoint,
vocabulary, input, masking, pooling, truncation, and code provenance. Embedding
files without this sidecar are legacy/unverified and must not be used for
corrected headline results.

Corrected probe configs must set `require_verified_embeddings: true`. The classifier
then rejects train/test embeddings produced by different checkpoints, dataset
manifests, or vocabularies and writes `provenance.json` beside `metrics.json`.

When training linear probes (e.g. for DNA-shape or EC level classification), compare your pretrained embeddings against:
1. **One-Hot Codon Identity vectors** (proves learning beyond local codon lookup).
2. **Randomly Initialized Model Embeddings** (proves that the pretraining phase, and not just the neural architecture, is responsible for the representation quality).

DNA-shape results must additionally use explicit packing/CDS metadata, grouped
folds, and centered local-sequence controls:

```bash
python -m scripts.eval_shape_baselines \
  --run_dir runs/<RUN_ID> --ckpt best.pt \
  --test_npz data/processed/global/<DATASET_ID>/test_bs256.npz \
  --manifest data/processed/global/<DATASET_ID>/manifest.json \
  --packing-metadata data/processed/global/<DATASET_ID>/test_packing.tsv \
  --cds-metadata data/processed/global/<DATASET_ID>/cds_meta.tsv \
  --group-by genome --output-prefix runs/<RUN_ID>/scores/shape_genome_grouped
```

The 5-mer and 7-mer controls use centered context and therefore constitute a
stronger shortcut control than the causal model's information access. Report
gene-grouped results only when genome metadata is unavailable.

---

## 5. Performance and Architectural Ablations

Before freezing data or starting corrected training, run the lifecycle preflight on
CPU and the target Apple Silicon host as documented in
[`TRAINING_PREFLIGHT.md`](TRAINING_PREFLIGHT.md). The MPS command must report
`actual_device: mps`; CPU fallback is not a pass.

Before merging changes that modify attention kernels (e.g., GQA, SDPA) or data loader structures:
1. Run the throughput benchmark to record step rates:
   ```bash
   python -m scripts.benchmark_training_speed --config configs/tiny_mps.yaml
   ```
2. Track peak RAM and VRAM footprint.
3. Validate that training loss/perplexity curves match baseline configurations to confirm mathematical equivalence.
4. **Never call `torch.mps.empty_cache()` inside the training step loop** unless recovering from an OOM error. Cache flushes trigger expensive synchronization barriers on Apple Silicon.
## 6. ProteinLM Shared Engine

`src.protein_lm.train_lm` now assembles `ProteinLMTask`, the shared accumulated-
backprop strategy, and `TrainingEngine`. Its CLI remains unchanged:

```bash
python -m src.protein_lm.train_lm --config configs/protein_lm/small.yaml
python -m src.protein_lm.train_lm --config configs/protein_lm/small.yaml \
  --run-id <run-id> --resume runs/protein_lm/<run-id>/checkpoints/last.pt
```

New checkpoints contain the versioned `engine`, `task`, `strategy`, `rng`, and
`metadata` namespaces and retain the legacy ProteinLM model, optimizer, scheduler,
epoch, and progress aliases. Existing unambiguous ProteinLM `last.pt` checkpoints
remain valid resume inputs.
