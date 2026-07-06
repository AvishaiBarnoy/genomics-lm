# Manual

This manual provides detailed information on how to use the `codonlm` and `proteinlm` modules.

## Genomics-LM (`codonlm`)

A compact codon‑level GPT‑style LM with a reproducible training + analysis pipeline.

### Quick Start

- Setup: conda env create -f environment.yml; conda activate codonlm (or pip install -r requirements.txt)
- Train (default config + auto RUN_ID):
  - ./main.sh
  - Outputs: checkpoints → runs/<RUN_ID>/checkpoints/, curves/metrics → runs/<RUN_ID>/scores/, full log → runs/<RUN_ID>/log.txt
- Analyze (one command):
  - ./analysis.sh <RUN_ID> [configs/tiny_mps.yaml]
- Query a trained model:
  - python -m scripts.query_model <RUN_ID> --mode next --dna ATGAAACCC --topk 5

### What’s In Here

- TinyGPT model (src/codonlm/model_tiny_gpt.py) with optional grad checkpointing and segment‑masked attention for <SEP>.
- Trainer with AMP, cosine warmup, early stopping, CSV curves (src/codonlm/train_codon_lm.py).
- Data prep that extracts CDS, tokenizes codons, builds NPZ datasets, and checks integrity (scripts/pipeline_prepare.py).
- Analysis scripts for frequencies, embeddings, attention, next‑token probes, saliency, and linear probes (scripts/*).

### Compare Runs

- Scan mode: python -m scripts.compare_runs  → outputs/scores/compare/summary.csv (ppl, params, and optional prefix‑generation metrics).

### Tips

- If data integrity fails (pad‑only windows), re‑run with --force or reduce block_size/windows_per_seq.
- On Apple Silicon, AMP is enabled; CE is computed in float32 to avoid NaNs.

### Training configs & toggles (advanced)

You can configure training runs using YAML files under `configs/`. Complete template configurations are available at [codon_lm_example.yaml](file:///Users/User/github/genomics-lm/configs/codon_lm_example.yaml) and [protein_lm_example.yaml](file:///Users/User/github/genomics-lm/configs/protein_lm_example.yaml). Supported keys include:

*   **Model Architecture:**
    *   `vocab_size`: Size of vocabulary (usually 69 for CodonLM, containing 64 codons and 5 specials).
    *   `block_size`: Maximum sequence length receptive window (e.g., 512).
    *   `n_layer`: Number of transformer blocks (e.g., 6 or 10).
    *   `n_head`: Number of self-attention heads (must divide `n_embd`).
    *   `n_embd`: Hidden state embedding dimension size (e.g., 256).
    *   `dropout`: Dropout probability applied to projections and embeddings.
    *   `label_smoothing`: Epsilon smoothing factor for CrossEntropyLoss.
    *   `use_checkpoint`: Enable activation gradient checkpointing (saves GPU RAM).
    *   `use_sdpa`: Use PyTorch's native Scaled Dot Product Attention for speedups.
    *   `n_kv_head` (GQA): Number of key-value heads for Grouped Query Attention (GQA).
    *   `tie_embeddings`: Share weights between token and output embeddings.
*   **Data Packing & Masking:**
    *   `pack_mode`: Dataset format, either `single` (one CDS per window) or `multi` (sequences packed).
    *   `sep_mask_enabled`: Enable attention masking across `<SEP>` boundaries in packed mode.
*   **Transfer Learning / Resumption / Freezing:**
    *   `transfer_from`: Path to pre-trained weights `.pt` file to initialize model parameters while discarding optimizer state.
    *   `freeze_backbone`: Set to `true` to freeze all transformer layers and standard prediction heads, training only the auxiliary projection heads.
*   **Optimizer & Scheduler:**
    *   `optimizer`: Select `"adamw"` or `"adafactor"` (reduces memory footprint).
    *   `lr`: Peak learning rate.
    *   `min_lr`: End-of-cycle minimum learning rate for cosine scheduler.
    *   `weight_decay`: L2 regularization decay strength.
    *   `warmup_steps`: Iterations of linear learning rate warmup.
*   **Training Loops:**
    *   `batch_size`: Physical batch size per device step.
    *   `grad_accum_steps`: Accumulate gradients over this many steps to simulate large batch sizes.
    *   `early_stop_patience`: Stop training after this many epochs without validation loss improvements.
    *   `max_time_minutes`: Limit training run duration. Saves checkpoint and exits gracefully if exceeded.
    *   `batch_optimizer`: Optional section for benchmarking `batch_size` / `grad_accum_steps` candidates before a long run.
*   **Output Directories:**
    *   `out_dir`: Location to save checkpoints.
    *   `scores_dir`: Location to save diagnostics.
*   **Look-Ahead Multi-Offset Projections (Stage 2.6+):**
    *   `multi_offset_loss_enabled`: Set to `true` to train auxiliary future-prediction heads.
    *   `multi_offset_targets`: List of integer offsets to predict (e.g. `[2, 4, 8, 16, 32]`).
    *   `multi_offset_weights`: Map of target offsets to their loss scaling factors (e.g. `2: 0.10, 4: 0.10`).

### Batch/Accumulation Optimizer

Before a long CodonLM run, benchmark physical batch size and gradient
accumulation settings on the real model/dataset:

```bash
python -m scripts.optimize_train_batching \
  --config configs/physical_termination_transfer.yaml \
  --run_id 2026-06-19_physical_termination_transfer_mps_b4_e1 \
  --benchmark
```

To benchmark and then automatically start/resume training with the fastest safe
setting:

```bash
caffeinate -i python -m scripts.optimize_train_batching \
  --config configs/physical_termination_transfer.yaml \
  --run_id 2026-06-19_physical_termination_transfer_mps_b4_e1 \
  --resume runs/2026-06-19_physical_termination_transfer_mps_b4_e1/checkpoints/last.pt \
  --optimize
```

For normal long runs, prefer the evolved `main.sh` training suite runner.

### Evolved Suite Runner (`main.sh`)
The `main.sh` script acts as the primary orchestration entrypoint for **data preprocessing and model training** (it does **NOT** handle downstream sequence generation/inference). It reads the `trainer` type from your YAML config (defaulting to `codon_lm` if not specified):

```yaml
# configs/my_config.yaml
trainer: codon_lm      # Options: codon_lm | protein_lm | protein_multitask | protein_classifier
```

#### Executing Runs:
```bash
# E.g., normal training execution
caffeinate -i ./main.sh --config configs/physical_termination_transfer.yaml

# Preprocess dataset and verify files without starting training
./main.sh --config configs/tiny_mps.yaml --preprocess-only

# Check trainer resolution and planned execution parameters without running commands
./main.sh --config configs/protein_critic.yaml --dry-run
```

If the trainer type is `codon_lm` and the config contains an enabled `batch_optimizer` block, `main.sh` routes training through `scripts.optimize_train_batching`; otherwise it calls `src.codonlm.train_codon_lm` directly. For any `protein` trainer type, dynamic codon prep and evaluations are automatically skipped. Cached optimizer results are reused unless `--force` is passed or `batch_optimizer.force: true` is set:

```bash
caffeinate -i ./main.sh \
  --config configs/physical_termination_transfer.yaml \
  --resume runs/2026-06-19_physical_termination_transfer_mps_b4_e1/checkpoints/last.pt
```

The same behavior can be configured in YAML:

```yaml
batch_optimizer:
  enabled: true
  mode: benchmark        # benchmark | optimize
  force: false           # true reruns benchmark even if cached results match
  include_current: true  # also benchmark top-level batch_size/grad_accum_steps
  candidates:
    - [2, 16]
    - [4, 16]
    - [4, 32]
    - [8, 16]
    - [8, 32]
  warmup_steps: 20
  measure_steps: 100
  force_gpu: true
```

Outputs are written to `runs/<RUN_ID>/scores/batch_optimizer/`. By default,
`include_current: true` prepends the top-level `batch_size` / `grad_accum_steps`
pair to the candidate list and deduplicates it, so the known manual setting is
always tested. Matching cached benchmark results are reused on later runs; set
`batch_optimizer.force: true` or pass `--force` to rerun the sweep. The optimizer
uses subprocesses for each candidate so OOM/allocation failures can be recorded
without aborting the whole benchmark. With `force_gpu: true`, it fails fast if
the run would fall back to CPU instead of MPS/CUDA.

When resuming mid-epoch, do not force a new sweep unless needed. If `--force`
selects a different `batch_size` or `grad_accum_steps` than the checkpoint used,
the trainer restores model/optimizer state but ignores the old mid-epoch skip
position and restarts the current epoch from batch 0. That is safe, but it
replays already-seen examples.

### Stage‑2 Classifiers

- Goal: benchmark sequence‑level representations from the LM against classical baselines.
- Extract embeddings from a run:
  - python -m scripts.extract_embeddings --run_id <RUN_ID> --fasta data/my.fasta --out runs/<RUN_ID>/scores/train_embeddings.npz
- Train a probe or baseline (configure paths in configs/classifier/*):
  - python -m scripts.train_classifier --config configs/classifier/probe_aa.yaml
  - python -m scripts.train_classifier --config configs/classifier/kmer_aa.yaml
- Evaluate a saved classifier:
  - python -m scripts.eval_classifier --kind probe --model runs/<RUN_ID>/scores/model.pkl --embeddings <NPZ> --labels <CSV> --out runs/<RUN_ID>/scores
- Protocols:
  - TSTR/TRTS are supported by choosing train_* and test_* sources in the config (e.g., synthetic vs real).

### Examples:

```bash
# Predict next codon probabilities
python -m scripts.infer_predict_next_codon --run_dir runs/<RUN_ID>/checkpoints --prompt "ATG GCT GCT" --topk 10

# Generate a CDS until a stop codon or EOS
python -m scripts.infer_generate_cds --run_dir runs/<RUN_ID>/checkpoints --stop_on_bio_stop --max_codons 300

# Score per-position ΔlogP for a provided CDS and plot a heatmap
python -m scripts.infer_score_mutations --run_dir runs/<RUN_ID>/checkpoints --seq "ATG GCT ... TGA" --out_dir runs/<RUN_ID>/scores
```

### Long Protein Generation (Prefix Benchmark)

- Run the prefix‑generation benchmark with long CDS targets:
  - python -m scripts.eval_generation_prefix --run_id <RUN_ID> --k_list 1,3,5,10 \
    --samples 5 --max_genes 50 --max_new 500 --min_aa_len 300 --target_aa_len 360 \
    --max_aa_len 400 --require_terminal_stop --special_margin 6
- Constraint: k + target_aa_len + special_margin ≤ block_size (from the model config). Lower target_aa_len or increase block_size if violated.
- Outputs add AA length stats (mean/median), terminal stop rate, hard‑cap rate, and an extra plot `aa_len_vs_k.png`.
- Runs trained with the termination auxiliary head can test decoder-side stop
  guidance:
  - python -m scripts.eval_generation_prefix --run_id <RUN_ID> --ckpt best.pt \
    --device mps --preset quick --termination_bias --termination_stop_bias 5.0 \
    --termination_trigger_class_max 4 --termination_bias_window 5
  The bias window gates stop pressure to the last N codons before the target
  length so stop guidance does not collapse into short peptides.

- **Generated-Prefix Replay Training**:
  To resolve the non-termination/hard-cap stalling failure mode:
  1.  **Generate Replay Data**: Sample prefix continuations from your pre-trained termination-aux checkpoint to capture hard-cap failure states and generate sparse distance-to-stop labels:
      ```bash
      python -m scripts.build_generated_prefix_replay \
        --run_id <RUN_ID> \
        --ckpt best.pt \
        --device cpu \
        --preset quick \
        --out runs/<RUN_ID>/scores/generated_prefix_replay.jsonl
      ```
  2.  **Fine-tune with Replay**: Configure your config YAML to enable joint replay training (mixing native dataset and prefix corrections):
      ```yaml
      replay_loss_enabled: true
      replay_loss_weight: 0.2
      replay_data: "runs/<RUN_ID>/scores/generated_prefix_replay.jsonl"
      ```
      Launch training:
      ```bash
      python -m src.codonlm.train_codon_lm --config configs/<REPLAY_CONFIG>.yaml --run_id <REPLAY_RUN_ID>
      ```
- Runs trained with multi-offset prior projection heads can evaluate look-ahead prior-guided decoding:
  - python -m scripts.eval_generation_prefix --run_id <RUN_ID> --ckpt best.pt \
    --device cpu --preset quick --multi_offset_prior \
    --multi_offset_prior_weights '{"2": 0.05, "4": 0.05, "8": 0.05, "16": 0.03, "32": 0.02}'
  The look-ahead weights bias decoding logits towards target structural features (e.g. helices at $x=4$ and sheets at $x=2$).

### Benchmarking & Evaluation

Evaluate a trained run on the held‑out test split and compute sanity KPIs:

```bash
# Test cross‑entropy and perplexity; updates runs/<RUN_ID>/scores/metrics.json
python -m scripts.evaluate_test --run_dir runs/<RUN_ID>/checkpoints

# Sanity KPIs (codon_corr, frameshift_delta, start/stop deltas, syn_gap)
python -m scripts.sanity_kpis --run_dir runs/<RUN_ID>/checkpoints

# Compare multiple runs and produce a table + plots
python -m scripts.compare_runs
# outputs:
#   runs/_summary/summary.csv
#   runs/_summary/ppl_vs_params.png
#   runs/_summary/val_vs_test_ppl.png
```

The benchmarking scripts merge results into each run’s `runs/<RUN_ID>/scores/metrics.json` without overwriting unrelated fields.

### Secondary-Structure Checks (optional)

- Heuristic propensities (unsupervised):
  - python -m scripts.ss_propensity --run_id <RUN_ID>
  - Or: python -m scripts.ss_propensity --dna data/processed/<name>/cds_dna.txt --out_dir outputs/analysis/ss_propensity
  - Writes per-sequence segment stats and length histograms; merges median helix/sheet segment lengths into metrics.json when run_id is used.
- Linear probe (supervised):
  - Prepare NPZ with token embeddings and per-token SS labels (H/E/C): H (N,T,D), Y (N,T), optional M (N,T).
  - python -m scripts.probe_ss_linear --emb_npz path/to/ss_tokens.npz --out_dir outputs/analysis/ss_probe
  - Reports accuracy, macro‑F1, AUROC, and a confusion matrix.
Notes: Propensity analysis is heuristic/correlation‑level. For stronger labels, use a local SS predictor (e.g., PSIPRED/NetSurfP) to generate H/E/C and then run the probe.

### Disorder Heuristics (optional)

- Estimate disorder signals complementary to SS:
  - python -m scripts.disorder_heuristics --run_id <RUN_ID>
  - Or: python -m scripts.disorder_heuristics --dna data/processed/<name>/cds_dna.txt --out_dir outputs/analysis/disorder
- Outputs: summary.csv with charge–hydropathy (Uversky) classification, disorder-promoting residue fraction, low-complexity segments; plots (CH-plane, length histograms). Merges a few aggregate KPIs into metrics.json when run_id is used.

### Sequence Quality & Calibration (optional)

- End-to-end verifier:
  - python -m scripts.seq_quality --run_id <RUN_ID>
  - Or: python -m scripts.seq_quality --dna data/processed/<name>/cds_dna.txt --ref_cds data/processed/<ref>/cds_dna.txt --ref_usage path/to/usage.tsv --ref_cai path/to/cai.tsv
  - Computes ORF integrity, length/GC%, codon usage KL/JS vs reference, CAI (if provided), FFT 1/3 periodicity, and diversity/novelty (k-mer Jaccard + MinHash). Merges headline KPIs into metrics.json.
- Calibration on a split:
  - python -m scripts.calibration_metrics --ckpt outputs/checkpoints/<RUN_ID>/best.pt --npz data/processed/combined/<RUN_ID>/val_bs512.npz --out outputs/scores/<RUN_ID>/metrics.json
  - Reports ECE and Brier score (PAD tokens ignored).

### Stage 2: Diversified Scaling & Transfer Learning

- **Transfer Learning**: fine-tune a model from pre-trained weights without carrying over optimizer state.
  - `python -m src.codonlm.train_codon_lm --config configs/stage2_diverse.yaml --transfer_from runs/<RUN_ID>/weights.pt`
- **Biological Motif Benchmark**: grade discovered patterns against real-world biological signals.
  - `python scripts/benchmark_motifs.py <RUN_ID>`
  - Result: `runs/<RUN_ID>/motif_mining/biological_benchmark.json` (Includes 'Biological Recall Score').
- **Plain English Summaries**: generate a human-readable interpretation of model results for non-experts.
  - `python scripts/generate_plain_english_report.py <RUN_ID>`
  - Result: `runs/<RUN_ID>/PLAIN_ENGLISH_SUMMARY.md`

## Protein Language Model (`protein_lm`)

### Overview

The `protein_lm` module is designed to model protein sequences and their functional properties. It consists of two main components:

1.  **`ProteinConditionalTransformer`**: A language model that learns to predict the next amino acid in a sequence, conditioned on functional or topological labels.
2.  **`ProteinClassifier`**: A classifier that uses the language model's architecture to predict the functional class of a given protein sequence.

### Tokenization

The tokenizer, located in `src/protein_lm/tokenizer.py`, is responsible for converting protein sequences and condition tokens into integer IDs that can be fed into the models.

-   **Vocabulary**: Includes 20 standard amino acids, an 'X' token for unknown residues, special tokens (`<BOS>`, `<EOS>`, `<PAD>`), and condition tokens (e.g., `<FUNC:ENZYME>`, `<TOPO:TM>`).
-   **Input Format**: The models expect input in the format `[BOS] + [condition_ids] + [sequence_ids]`.

### Configuration

Model architecture, training parameters, and data paths are defined in YAML configuration files located in `configs/protein_lm/`.

-   `small.yaml`: A sample configuration for training the language model.
-   `classifier_small.yaml`: A sample configuration for training the classifier. This file is similar to `small.yaml` but includes a `num_classes` parameter for the classification head.

### Training

The module includes two training scripts in `src/protein_lm/`.

#### Language Model Training

To train the language model, run the `train_lm.py` script with a configuration file:

```bash
python -m src.protein_lm.train_lm --config configs/protein_lm/small.yaml
```

The script will:
1.  Load the model configuration and training parameters.
2.  Initialize the `ProteinConditionalTransformer` model.
3.  Load the training and validation data using the `ProteinDataset`.
4.  Train the model using cross-entropy loss to predict the next token.
5.  Save checkpoints to `outputs/protein_lm/<run_id>/`.

#### Classifier Training

To train the classifier, run the `train_classifier.py` script:

```bash
python -m src.protein_lm.train_classifier --config configs/protein_lm/classifier_small.yaml
```

This script will:
1.  Load the classifier configuration.
2.  Initialize the `ProteinClassifier` model.
3.  Load the training and validation data using the `ProteinClassificationDataset`, which is designed to handle class labels.
4.  Train the model using cross-entropy loss for classification.
5.  Log validation accuracy and F1 score.
6.  Save checkpoints to `outputs/protein_classifier/<run_id>/`.

#### Multi-Task Protein Critic Training

The Multi-Task Protein Critic backbone (`src/protein_lm/train_multi_task.py`) trains a single model to simultaneously predict Pfam Family, EC Function Number, and Thermodynamic Stability.

To train the Protein Critic, run:

```bash
python -m src.protein_lm.train_multi_task --config configs/protein_critic.yaml
```

**Training Specs:**
- **Architecture:** 8 layers, 8 attention heads, 256 embedding dimension (`8L8H_d256`).
- **Resilience:** Saves epoch-end training states to `outputs/checkpoints/protein_critic/last_critic.pt` and supports resumption via `--resume`.
- **Outputs:** Saves the best evaluation checkpoint to `outputs/checkpoints/protein_critic/best_critic.pt`.

**Current Top Model Performance:**

| Task | Target Type | Number of Classes | Samples Evaluated | Model Accuracy | Random Baseline | Improvement Factor |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Stability** | Physical (MegaScale stability) | 2 | 138 | **76.81%** | 50.00% | **1.5x** |
| **Family** | Identity (Pfam ID) | 1,000 | 553 | **6.15%** | 0.10% | **61.5x** |
| **Function** | Biological (EC Number ID) | 500 | 200 | **5.50%** | 0.20% | **27.5x** |

### Testing

The `protein_lm` module comes with a suite of tests to verify its functionality. To run the tests, use `pytest`:

```bash
pytest tests/test_protein_tokenizer.py
pytest tests/test_protein_models.py
```

### Extending the Module

#### Adding New Condition Tokens

The `ProteinTokenizer` can be easily extended to support new condition tokens. To add a new token, you need to modify the `condition_tokens` dictionary in `src/protein_lm/tokenizer.py`.

For example, to add a new condition for a subcellular location, you could do the following:

```python
# In src/protein_lm/tokenizer.py

self.condition_tokens = {
    'FUNC_ENZYME': '<FUNC:ENZYME>',
    'FUNC_NON_ENZYME': '<FUNC:NON_ENZYME>',
    'TOPO_TM': '<TOPO:TM>',
    'TOPO_GLOBULAR': '<TOPO:GLOBULAR>',
    'LOC_MEMBRANE': '<LOC:MEMBRANE>', # New condition token
}
```

The tokenizer will automatically update its vocabulary and token-to-ID mappings. You will also need to update your data preparation scripts to include the new labels in the JSONL files.

## Web Dashboard & Model Playground

The repository includes a Streamlit-based web dashboard that provides an interactive visualization interface for model runs, metrics, attention mapping, embedding exploration, and a live Model Playground (where you can run next-codon queries and sequence generation supervised on-the-fly by the multi-task Protein Critic).

### Running the Dashboard

To launch the local Streamlit dashboard, run the following command from the repository root:

```bash
streamlit run scripts/web_dashboard.py
```

By default, the client will start a local server (typically at `http://localhost:8501`).

### Features

- **🏠 Run Overview & Metrics**: Browse all completed runs under `outputs/checkpoints/` and visualize their training/validation metrics.
- **🧬 Saliency & Mutation Maps**: Audit model predictions and generate heatmaps of mutation probabilities.
- **🧪 Model Playground**: Run causal generation queries on the CodonLM model (adjusting Temperature, Top-K, and Max Codons) and instantly translate the resulting DNA to amino acids.
- **🛡️ Protein Critic Bridge**: The playground automatically translates generated DNA sequences and scores their predicted Pfam family, EC number function, and thermodynamic stability using the loaded Protein Critic model.

## SOTA Benchmarking & Compute Footprint Profiling

To evaluate our model against published SOTA prokaryotic models (Evo 1, GenSLM) on aligned prokaryotic benchmarks:

1.  **Prepare mock/synthetic benchmark datasets:**
    ```bash
    python scripts/prepare_sota_benchmarks.py
    ```
    This creates datasets under `data/benchmarks/` for zero-shot mutation scoring and gene essentiality.

2.  **Evaluate Zero-Shot Mutational Fitness:**
    ```bash
    python -m scripts.benchmark_zero_shot_mutations --run_id <RUN_ID>
    ```
    This computes sequence log-likelihood comparisons on protein and rRNA DMS datasets and writes Spearman rank correlations to the run's `metrics.json`.

3.  **Evaluate Gene Essentiality (Linear Probes):**
    ```bash
    python -m scripts.benchmark_gene_essentiality --run_id <RUN_ID>
    ```
    This trains stratified 5-fold cross-validated logistic regression probes on the mean-pooled sequence embeddings of Lambda phage and *P. aeruginosa* genes.

4.  **Generate SOTA Comparison and Compute Footprint Report:**
    ```bash
    python -m scripts.generate_sota_report --run_id <RUN_ID>
    ```
    This generates a comprehensive markdown comparison report (`SOTA_BENCHMARK_REPORT.md` inside the run directory) containing parameter and resource efficiency density ratios.

---

## Downstream Classification Probes

The repository supports two types of downstream classifiers trained on top of frozen LM embeddings:

### LM Embedding Probes

Extract mean-pooled hidden states from a trained run, then train linear or MLP classifiers:

```bash
# Step 1: Extract embeddings
python -m scripts.extract_embeddings \
  --run_id <RUN_ID> \
  --csv data/processed/ec_train_seqs.csv \
  --seq_col seq \
  --mode codon_tokens \
  --out outputs/reports/ec_classification/train_embeddings.npz

# Step 2: Train a probe
python -m scripts.train_classifier --config configs/classifier/probe_ec.yaml
python -m scripts.train_classifier --config configs/classifier/probe_amr.yaml
```

Available probe configs in `configs/classifier/`:

| Config | Task | Classifier |
|---|---|---|
| `probe_ec.yaml` | EC Level-1 classification | Logistic Regression |
| `probe_amr.yaml` | AMR antibiotic class (CARD v3) | Logistic Regression |
| `probe_amr_svm.yaml` | AMR antibiotic class (CARD v3) | Linear SVM |
| `probe_ec_mlp.yaml` | EC Level-1 classification | MLP 2×128 |

### K-mer Baselines

Train raw 3-mer TF-IDF logistic regression on the same train/test splits (no LM required):

```bash
python -m scripts.train_classifier --config configs/classifier/kmer_ec.yaml
python -m scripts.train_classifier --config configs/classifier/kmer_amr.yaml
```

### AMR Dataset Preparation

Download and process CARD (Comprehensive Antibiotic Resistance Database, CC BY 4.0):

```bash
python -m scripts.prepare_amr_dataset
# Outputs: data/labels/train_amr.csv, data/labels/test_amr.csv, data/labels/amr_label_map.json
# Also: data/processed/train_amr_seqs.csv, data/processed/test_amr_seqs.csv (for k-mer)
```

---

## Conference Figure Generation

Generate publication-quality dark-background figures from a trained run's `artifacts.npz`:

```bash
# Figure 1: UMAP codon embedding space (synonymous codon clustering)
python -m scripts.conference_umap <RUN_ID>

# Figure 2: Attention head specialization heatmaps
python -m scripts.conference_attention <RUN_ID>
```

Both scripts auto-copy outputs to `conference/figures/`. Scientific interpretations for each panel are in `conference/figure_descriptions.md`.

---

## Storage Layout & `outputs/` vs `runs/`

The repository uses two output directory conventions due to a historical migration:

| Directory | Purpose | Status |
|---|---|---|
| `runs/<RUN_ID>/` | **Primary active store** — checkpoints, logs, scores, charts, artifacts.npz | ✅ Current |
| `outputs/checkpoints/` | **Legacy** — Stage 1 checkpoints from before the `runs/` layout | ⚠️ Safe to archive |
| `outputs/reports/` | Downstream probe outputs (EC, AMR, k-mer classifiers) | ✅ Current |
| `outputs/scores/` | Legacy benchmark scores | ⚠️ Historical |
| `conference/` | Publication assets — figures, SOTA table, abstracts, slides | ✅ Current |

> **Why is `outputs/` so large?**
> `outputs/checkpoints/` (~4.4 GB) contains 31 old Stage 1 and early Stage 2 model checkpoints that predate the migration to the `runs/` layout. They are not used by any current script and can be safely deleted or moved to cold storage:
> ```bash
> # Archive to external drive or just remove:
> rm -rf outputs/checkpoints/
> ```
> All active model checkpoints are in `runs/<RUN_ID>/checkpoints/`.
