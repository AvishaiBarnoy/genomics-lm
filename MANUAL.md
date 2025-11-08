Genomics‑LM Manual

Overview

- Purpose: Train and analyze a codon‑level GPT‑style LM for prokaryotic CDS with a 6‑step interpretability pipeline.
- Design: Simple data pipeline → compact TinyGPT model → robust training with AMP/checkpointing → analysis scripts that write tables/plots into runs/<RUN_ID>.

Quick Start

- Environment: conda activate codonlm
- One‑shot pipeline: ./main.sh -c configs/tiny_mps.yaml
- Inspect logs: runs/<RUN_ID>/log.txt, outputs/checkpoints/<RUN_ID>/best.pt, outputs/scores/<RUN_ID>
- Post‑analysis extras: ./analysis.sh (motifs, artifacts) or post_process.sh

Concepts

- Tokens: 4 specials <PAD>, <BOS_CDS>, <EOS_CDS>, <SEP> + 64 codons.
- Packing: Multi‑CDS windows with <SEP> separators (pack_mode=multi) and block‑diagonal attention masking (sep_mask_enabled).
- Run ID: Auto‑generated like YYYY‑MM‑DD_tiny_6L4H_d256_e5 via scripts/make_run_id.py; used to name outputs.

Config Schema (configs/tiny_mps.yaml)

- Model: vocab_size, block_size, n_layer, n_head, n_embd, dropout, sep_mask_enabled
- Data: datasets[{name,gbff,min_len}], windows_per_seq, val_frac, test_frac, pack_mode
- Train: batch_size, grad_accum_steps, lr, weight_decay, epochs|"auto", optimizer (adamw|adafactor), amp, use_checkpoint, label_smoothing
- Schedule: scheduler (cosine|plateau), warmup_steps, min_lr, plateau_patience, early_stop_patience
- System: compile, compile_mode, matmul_precision, num_workers, pin_memory, prefetch_factor, persistent_workers
- Outputs: out_dir (checkpoints), scores_dir, log_csv, seed, itos_path, saliency_window, saliency_top

Pipelines

- Data prep: scripts/pipeline_prepare.py reads config; extracts CDS, tokenizes, builds NPZs, concatenates across datasets, and writes runs/<RUN_ID>/pipeline_prepare.json + integrity.json.
- Training: src/codonlm/train_codon_lm.py trains with AMP (MPS aware), AdamW by default, cosine warmup schedule, early stopping, CSV curves.
- Evaluation: scripts/evaluate_test.py, scripts/probe_next_token.py, scripts/probe_linear.py, scripts/analyze_* write tables/plots in runs/<RUN_ID>.
- End‑to‑end: main.sh wraps all steps with logging, timing, and optional artifacts/motifs.

Benchmarks & Inference

- Perplexity: src/codonlm/eval_perplexity.py
- Prefix generation: scripts/eval_generation_prefix.py → outputs/scores/<RUN_ID>/gen_prefix/{samples.csv,summary.csv}
- Interactive/query: python -m scripts.query_model <RUN_ID> --mode next|generate|score --dna ATG...

Stage‑2 Classifiers (sequence‑level benchmarking)

- Extract embeddings from a trained run:
  - python -m scripts.extract_embeddings --run_id <RUN_ID> --fasta data/my.fasta --out outputs/reports/e1/train_embeddings.npz
  - Accepts --csv with a seq column; tokenizes DNA into codons and mean‑pools final hidden states (excludes PADs).
- Train/evaluate classifiers via YAML config:
  - python -m scripts.train_classifier --config configs/classifier/probe_aa.yaml
  - Supports: probe_logreg, probe_svm, mlp (on embeddings), and kmer_logreg/kmer_svm/kmer_xgb baselines on sequences.
- Evaluate a saved model on new data:
  - python -m scripts.eval_classifier --kind probe --model outputs/reports/e1/model.pkl --embeddings <NPZ> --labels <CSV> --out outputs/reports/e1
  - For k‑mer models, add --vectorizer and --seqs.
 - Protocols (benchmark setups):
   - TSTR (Train on Synthetic, Test on Real): point train_* to synthetic and test_* to real in the config.
   - TRTS (Train on Real, Test on Synthetic): reverse the sources.
   - The pipeline simply follows your declared paths; metrics are written to out_dir (metrics.json, confusion.png, calibration.png).

Data Integrity & Troubleshooting

- Integrity check: After prep, integrity.json counts pad‑only windows; if non‑zero, pipeline exits with instructions to use --force or adjust block_size/windows_per_seq.
- NaNs / non‑finite loss: Model computes CE in float32 even under AMP; if NaNs persist, re‑prep with --force, reduce label_smoothing, or set amp: false.
- MPS autocast: If autocast is unsupported, training falls back automatically without crashing.

Performance Tuning

- Apple Silicon: matmul_precision: high; AMP enabled; use_checkpoint may reduce memory at some runtime cost.
- Throughput: Increase batch_size or num_workers; prefer prefetch_factor/persistent_workers for >0 workers.
- Tokens per parameter: epochs: "auto" uses tokens_per_param × n_params heuristic; log prints derivation.

Logging & Provenance

- All stdout/stderr is tee’d into runs/<RUN_ID>/log.txt. Hardware, config snapshot (and sha256), git commit, dataset manifest, timing per epoch and total are logged.
- Training curves CSV saved under outputs/scores/<RUN_ID>/curves.csv with columns: step,train_loss,val_loss,perplexity,lr.

FAQ

- Where is early stopping? In train_codon_lm.py; compares val_loss per epoch; stops after early_stop_patience epochs without improvement. best.pt is only saved when val improves.
- Why no best.pt? If val never improves from +inf (e.g., NaNs) best.pt is not written; fix data integrity or training stability first.
- How to resume? main.sh -r outputs/checkpoints/<RUN_ID>/last.pt or python -m src.codonlm.train_codon_lm --resume ...

A/B Switches

- pack_mode: multi|single (window packing strategy)
- sep_mask_enabled: true|false (block cross‑ORF attention)
- optimizer: adamw|adafactor
- scheduler: cosine|plateau (cosine uses warmup + decay to min_lr)
