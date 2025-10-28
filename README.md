# Genomics-LM

A lightweight transformer language model for genomic sequences (codon-level).  
The goal is to explore embeddings, motif discovery, and sequence prediction with a minimal yet extensible architecture.

---

## Features
- ⚡ Tiny GPT-like model (configurable layers, heads, embedding size).
- 🧬 Works with DNA/codon tokenization.
- 🔍 Motif mining utilities.
- 🧪 Reproducible training with configs in YAML.
- ✅ Pytest test suite.

---

## Installation

Clone the repository and create a conda environment:

```bash
git clone https://github.com/AvishaiBarnoy/genomics-lm.git
cd genomics-lm
conda env create -f env/conda-environment.yml
conda activate codonlm
```

## Training

Training runs now track outputs per run id. Pick a `RUN_ID` (or export `RUN_ID=...`) and pass it on the CLI:

```bash
RUN_ID=$(date +"%Y%m%d-%H%M%S")  # or any unique label
python -m src.codonlm.train_codon_lm --config configs/tiny_mps.yaml --run_id "$RUN_ID"
```

Checkpoints land in `outputs/checkpoints/<RUN_ID>/` (`best.pt`, `last.pt`) and metrics live in `outputs/scores/<RUN_ID>/` (`metrics.json`, `curves.csv`). The shell pipeline (`main.sh`) sets a sensible default `RUN_ID` automatically, so you can still run:

```bash
chmod +x main.sh analysis.sh post_process.sh
./main.sh            # uses configs/tiny_mps.yaml by default
./main.sh -c path/to/other.yaml                     # supply your own config
./main.sh --dataset ecoli2,data/raw/GCF_xx.gbff     # add an extra genome ad-hoc
./main.sh --force                                   # rebuild cached processed data
./main.sh -r outputs/checkpoints/<RUN_ID>/best.pt   # resume training
./main.sh -c my.yaml -r outputs/checkpoints/<RUN_ID>/best.pt
```

### Dataset configuration

Multi-genome training is driven by the YAML config. A minimal example:

```yaml
datasets:
  - name: ecoli_k12
    gbff: data/raw/GCF_000005845.2_ASM584v2_genomic.gbff
    min_len: 90

block_size: 256
windows_per_seq: 2
val_frac: 0.1
test_frac: 0.1
```

Each dataset is cached under `data/processed/<name>/` (with DNA, metadata, token IDs, and NPZ splits). The pipeline automatically skips extraction/tokenization/build steps when the outputs already exist; use `--force` to rebuild. All datasets are concatenated into `data/processed/combined/<RUN_ID>/train|val|test_bs*.npz` before training so the trainer only needs one combined manifest.

Each run also records the dataset lineage in `runs/<RUN_ID>/combined_manifest.json`, which lists the per-genome token files (`itos.txt`) so downstream scripts can recover the vocabulary even when multiple genomes are mixed.

You can append extra datasets from the CLI without touching the config:

```bash
./main.sh --dataset ecoli_o157,data/raw/GCF_000008865.2_ASM886v2_genomic.gbff,120
```

## Post-Training Analysis Suite

The analysis suite includes:

- Token frequencies and first-token distribution
- Embedding PCA + nearest neighbors
- Attention heatmaps
- Next-token probes (basic predictive sanity checks)
- Saliency maps (+ a short motif spotlight)
- Linear probes with biology-aware labels (requires `probe_labels.csv`)

Quick start (recommended):

```bash
# Run all analyses for a trained run
./analysis.sh <RUN_ID> [configs/tiny_mps.yaml]
```

Or invoke individual steps:

```bash
RUN_ID=2025-09-30_tiny_2L4H_d128_e5
python -m scripts.collect_artifacts_yaml $RUN_ID configs/tiny_mps.yaml
python -m scripts.analyze_frequencies   $RUN_ID
python -m scripts.analyze_embeddings    $RUN_ID
python -m scripts.analyze_attention     $RUN_ID
python -m scripts.probe_next_token      $RUN_ID
python -m scripts.analyze_saliency      $RUN_ID
python -m scripts.report_top_saliency   $RUN_ID --window 9 --top 20

# Linear probes require token labels
python -m scripts.generate_probe_labels $RUN_ID
python -m scripts.probe_linear          $RUN_ID

# Optional: summarize and compare
python -m scripts.summarize_one_cds     $RUN_ID
python -m scripts.compare_runs          $RUN_ID <other_run_ids...>
```

Default trainer (AMP + cosine schedule + optional label smoothing):

```bash
python -m src.codonlm.train_codon_lm --config configs/tiny_mps.yaml --run_id "$RUN_ID"
# resume from a checkpoint
python -m src.codonlm.train_codon_lm --config configs/tiny_mps.yaml --resume outputs/checkpoints/$RUN_ID/best.pt --run_id "$RUN_ID"
# override datasets manually (comma-separated lists)
python -m src.codonlm.train_codon_lm --config configs/tiny_mps.yaml \
  --train_npz data/processed/combined/${RUN_ID}/train_bs256.npz \
  --val_npz data/processed/combined/${RUN_ID}/val_bs256.npz \
  --test_npz data/processed/combined/${RUN_ID}/test_bs256.npz \
  --run_id "$RUN_ID"
```

Note: cosine+warmup is enabled by default and you can set `label_smoothing: 0.05` in the YAML to improve probability calibration (reduces overconfident spikes in next-token predictions).

Convenience wrappers:

```bash
# artifacts only (organize weights, sample logits, etc.)
./post_process.sh <RUN_ID> [CONFIG]

# full 6-step analysis
./analysis.sh <RUN_ID> [CONFIG]

# optional exploratory EDA bundle
python -m scripts.run_eda <RUN_ID>
```

`analysis.sh` drives the maintained CLI modules under `scripts/`. The legacy `analysis/` folder holds optional exploratory helpers; the `scripts.run_eda` wrapper invokes them and saves outputs to `runs/<RUN_ID>/analysis/eda`.

For Step 6 linear probes:

```bash
python -m scripts.generate_probe_labels $RUN_ID
python -m scripts.probe_linear $RUN_ID
python -m scripts.export_run_summary $RUN_ID
```

Saliency/motif spotlight:

```bash
python -m scripts.report_top_saliency $RUN_ID --window 9 --top 20
```
