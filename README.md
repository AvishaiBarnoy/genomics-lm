# Genomics‑LM

A compact codon‑level GPT‑style LM with a reproducible training + analysis pipeline.

Quick Start

- Setup: conda env create -f env/conda-environment.yml; conda activate codonlm
- Train (default config + auto RUN_ID):
  - ./main.sh
  - Outputs: checkpoints → outputs/checkpoints/<RUN_ID>/, curves/metrics → outputs/scores/<RUN_ID>/, full log → runs/<RUN_ID>/log.txt
- Analyze (one command):
  - ./analysis.sh <RUN_ID> [configs/tiny_mps.yaml]
- Query a trained model:
  - python -m scripts.query_model <RUN_ID> --mode next --dna ATGAAACCC --topk 5

What’s In Here

- TinyGPT model (src/codonlm/model_tiny_gpt.py) with optional grad checkpointing and segment‑masked attention for <SEP>.
- Trainer with AMP, cosine warmup, early stopping, CSV curves (src/codonlm/train_codon_lm.py).
- Data prep that extracts CDS, tokenizes codons, builds NPZ datasets, and checks integrity (scripts/pipeline_prepare.py).
- Analysis scripts for frequencies, embeddings, attention, next‑token probes, saliency, and linear probes (scripts/*; see MANUAL).

Documentation

- MANUAL.md — full configuration, integrity checks, training details, and the complete analysis suite.
- ROADMAP.md — staged plan and progress notes.

Compare Runs

- Scan mode: python -m scripts.compare_runs  → outputs/scores/compare/summary.csv (ppl, params, and optional prefix‑generation metrics).

Tips

- If data integrity fails (pad‑only windows), re‑run with --force or reduce block_size/windows_per_seq; see MANUAL.md.
- On Apple Silicon, AMP is enabled; CE is computed in float32 to avoid NaNs.

Examples:

```bash
# Predict next codon probabilities
python -m scripts.infer_predict_next_codon --run_dir outputs/checkpoints/<RUN_ID> --prompt "ATG GCT GCT" --topk 10

# Generate a CDS until a stop codon or EOS
python -m scripts.infer_generate_cds --run_dir outputs/checkpoints/<RUN_ID> --stop_on_bio_stop --max_codons 300

# Score per-position ΔlogP for a provided CDS and plot a heatmap
python -m scripts.infer_score_mutations --run_dir outputs/checkpoints/<RUN_ID> --seq "ATG GCT ... TGA" --out_dir outputs/analysis/<RUN_ID>
```
## Benchmarking & Evaluation

Evaluate a trained run on the held‑out test split and compute sanity KPIs:

```bash
# Test cross‑entropy and perplexity; updates outputs/scores/<RUN_ID>/metrics.json
python -m scripts.evaluate_test --run_dir outputs/checkpoints/<RUN_ID>

# Sanity KPIs (codon_corr, frameshift_delta, start/stop deltas, syn_gap)
python -m scripts.sanity_kpis --run_dir outputs/checkpoints/<RUN_ID>

# Compare multiple runs and produce a table + plots
python -m scripts.compare_runs
# outputs:
#   outputs/scores/compare/summary.csv
#   outputs/scores/compare/ppl_vs_params.png
#   outputs/scores/compare/val_vs_test_ppl.png
```

The benchmarking scripts merge results into each run’s `outputs/scores/<RUN_ID>/metrics.json` without overwriting unrelated fields.
