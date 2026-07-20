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
   `leakage_audit.json`. Cross-split exact CDS duplicates or protein clusters are
   fatal.

   The default protein-cluster gate uses 30% sequence identity and 80% coverage.
   Override those recorded thresholds with `max_cross_split_protein_identity` and
   `min_homology_coverage` in the run config. `--skip-homology-audit` and
   `--allow-cross-split-exact-duplicates` exist only for fixtures and legacy
   reproduction; either marks the resulting manifest as non-scientific.

   Scientific preparation also fails when fewer than three groups are available.
   `--allow-sequence-split` is an explicit development-only escape hatch and marks
   the resulting manifest as non-scientific. The older
   `scripts.pipeline_prepare` route is disabled unless
   `--allow-legacy-per-dataset-split` is supplied for historical reproduction.
3. Verify that the output splits contain mutually exclusive genome sets by inspecting the generated `data/processed/global/<RUN_ID>/cds_meta.tsv`.

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

---

## 3. Synonymous and Shuffling Controls

### The Golden Rule
**Verify if the model learns codon-level biophysics or just amino acid maps.**

### Mandatory Workflow
1. Generate control datasets from your test split:
   ```bash
   python -m scripts.generate_synonymous_controls --test_npz data/processed/global/<RUN_ID>/test_bs256.npz
   ```
   This outputs:
   - `test_control_synonymous_bs256.npz` (random synonymous codons, same protein).
   - `test_control_codon_shuffle_bs256.npz` (shuffled codons, same composition).
   - `test_control_protein_shuffle_bs256.npz` (shuffled protein, same codon counts).
2. Run [`evaluate_test.py`](file:///Users/User/github/genomics-lm/scripts/evaluate_test.py) on each of these control NPZs.
3. Assert that the model's perplexity is significantly better (lower) on the natural test set compared to the synonymous recoding set.

---

## 4. Downstream Probe Controls

### The Golden Rule
**Always control for token-identity decodability in shape and classification probes.**

### Mandatory Workflow
When training linear probes (e.g. for DNA-shape or EC level classification), compare your pretrained embeddings against:
1. **One-Hot Codon Identity vectors** (proves learning beyond local codon lookup).
2. **Randomly Initialized Model Embeddings** (proves that the pretraining phase, and not just the neural architecture, is responsible for the representation quality).

---

## 5. Performance and Architectural Ablations

Before merging changes that modify attention kernels (e.g., GQA, SDPA) or data loader structures:
1. Run the throughput benchmark to record step rates:
   ```bash
   python -m scripts.benchmark_training_speed --config configs/tiny_mps.yaml
   ```
2. Track peak RAM and VRAM footprint.
3. Validate that training loss/perplexity curves match baseline configurations to confirm mathematical equivalence.
4. **Never call `torch.mps.empty_cache()` inside the training step loop** unless recovering from an OOM error. Cache flushes trigger expensive synchronization barriers on Apple Silicon.
