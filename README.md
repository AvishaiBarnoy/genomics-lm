# Genomics‑LM

A compact codon‑level GPT‑style LM with a reproducible training + analysis pipeline.

## Quick Start

- Setup: conda env create -f environment.yml; conda activate codonlm (or pip install -r requirements.txt)
- Train (default config + auto RUN_ID):
  - ./main.sh
- Analyze (one command):
  - ./analysis.sh <RUN_ID> [configs/tiny_mps.yaml]
- Query a trained model:
  - python -m scripts.query_model <RUN_ID> --mode next --dna ATGAAACCC --topk 5
- Run Web Dashboard & Playground:
  - streamlit run scripts/web_dashboard.py


## Documentation

- [ROADMAP.md](ROADMAP.md) — staged plan and progress notes.
- [docs/MANUAL.md](docs/MANUAL.md) — full configuration, integrity checks, training details, and the complete analysis suite for both `codonlm` and `proteinlm`.
- [docs/DEVELOPMENT_LOG.md](docs/DEVELOPMENT_LOG.md) — A comprehensive narrative of the project's evolution, detailing biological intuition, hardware optimization (M2 8GB), and the architectural progression from isolated genes to Operon-Aware Genomic Tapes and Hierarchical Protein-Critics.
- [docs/RELEASE_NOTES.md](docs/RELEASE_NOTES.md) — high‑level changes and new features across releases.
- [docs/agents.md](docs/agents.md) — overview of the logical agents making up the training pipeline.