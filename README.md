# Genomics‑LM

A compact codon‑level causal GPT for bacterial genomes, with downstream
evaluation and interpretability tooling.

> **Scientific status:** Published Stage 2.6 metrics use a legacy protocol that
> predates mandatory global genome-aware splitting and corrected causal embedding
> extraction. They are retained for historical auditability, not presented as
> current validated results. Leakage-controlled revalidation is tracked in
> [issue #92](https://github.com/AvishaiBarnoy/genomics-lm/issues/92).

---

## Quick Start

```bash
conda env create -f environment.yml && conda activate codonlm
./main.sh                                             # train with default config
./analysis.sh <RUN_ID>                                # full analysis suite
python -m scripts.query_model <RUN_ID> --mode next --dna ATGAAACCC --topk 5
streamlit run scripts/web_dashboard.py                # interactive dashboard
```

## Development Checks

Run the same core checks required by GitHub Actions:

```bash
ruff check . --select E9,F63,F7,F82
pytest -q -m "not slow and not external and not mps" --cov=src --cov=scripts --cov-report=term
git status --short
```

The required CPU suite runs on the supported Python 3.11 environment. Tests marked `slow`,
`external`, or `mps` are opt-in and must not be used to claim CPU CI coverage.
MPS performance benchmarks must be run manually on Apple Silicon using the
documented benchmark commands.

## Downstream Evaluation

```bash
# Enzyme Commission (EC) classification probe
python -m scripts.extract_embeddings --run_id <RUN_ID> --csv data/processed/ec_train_seqs.csv --mode codon_tokens --out outputs/reports/ec_classification/train_embeddings.npz
python -m scripts.train_classifier --config configs/classifier/probe_ec.yaml

# AMR classification probe (CARD v3)
python -m scripts.prepare_amr_dataset                 # downloads & prepares CARD data
python -m scripts.train_classifier --config configs/classifier/probe_amr.yaml

# K-mer baselines (EC + AMR)
python -m scripts.train_classifier --config configs/classifier/kmer_ec.yaml
python -m scripts.train_classifier --config configs/classifier/kmer_amr.yaml

# DNAshape regression probe
python -m scripts.probe_structural_regression --run_id <RUN_ID>
```

## Conference Figures

```bash
python -m scripts.conference_umap <RUN_ID>            # UMAP codon embedding plot
python -m scripts.conference_attention <RUN_ID>       # Attention head specialization
```
Outputs land in `conference/figures/`. All assets and the SOTA table are in [`conference/`](conference/).

## Storage Layout

| Directory | Purpose |
|---|---|
| `runs/<RUN_ID>/` | **Primary** — checkpoints, logs, scores, charts, artifacts |
| `outputs/checkpoints/` | **Legacy** — Stage 1 checkpoints pre-dating `runs/` migration (safe to archive) |
| `outputs/reports/` | Downstream probe reports (EC, AMR, k-mer) |
| `conference/` | Publication-ready figures, SOTA table, abstracts |
| `data/raw/`, `data/processed/`, `data/labels/` | Training and evaluation data |

> ⚠️ `outputs/checkpoints/` (~4.4 GB) contains old Stage 1 run checkpoints that predate the `runs/` layout. All active models live in `runs/`. The `outputs/checkpoints/` directory can be safely archived or deleted.

## Documentation

- [ROADMAP.md](ROADMAP.md) — staged plan and progress notes
- [conductor/tracks.md](conductor/tracks.md) — all research tracks and completion status
- [docs/MANUAL.md](docs/MANUAL.md) — full configuration, training, and analysis reference
- [docs/DEVELOPMENT_LOG.md](docs/DEVELOPMENT_LOG.md) — narrative project history
- [docs/RELEASE_NOTES.md](docs/RELEASE_NOTES.md) — high-level changelog
- [conference/sota_benchmark_table.md](conference/sota_benchmark_table.md) — historical benchmark table with protocol-status labels
