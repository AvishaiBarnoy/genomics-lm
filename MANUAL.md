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
- Reference tables: scripts/build_reference_tables.py builds per‑organism codon usage and CAI weights under data/reference/<name>/ from CDS, avoiding duplicate files.

Benchmarks & Inference

- Perplexity: src/codonlm/eval_perplexity.py
- Prefix generation: scripts/eval_generation_prefix.py → outputs/scores/<RUN_ID>/gen_prefix/{samples.csv,summary.csv}
- Interactive/query: python -m scripts.query_model <RUN_ID> --mode next|generate|score --dna ATG...

Long CDS generation (prefix benchmark)

- Goal: evaluate longer protein generation (300–400 aa) while respecting model block_size.
- CLI flags in scripts/eval_generation_prefix.py:
  - --min_aa_len (default 100), --target_aa_len (default 256), --max_aa_len (default 400)
  - --require_terminal_stop (enforce canonical stop at the end)
  - --special_margin (default 6) for specials like BOS/EOS/SEP
- Constraint: k + target_aa_len + special_margin ≤ block_size. If violated, lower target_aa_len or increase block_size.
- Generation logic (src/codonlm/generate.py):
  - Generates up to a hard_cap = min(block_size − k − special_margin, max_aa_len, max_new).
  - Stops when a canonical stop is reached at/after target length, or at target length if not requiring a terminal stop, or at hard_cap.
  - Records had_terminal_stop, early_stop (internal stop before target), hit_hard_cap, and gen_len_codons.
- Outputs:
  - samples.csv includes gen_len_codons, had_terminal_stop, hit_hard_cap, target_codons.
  - summary.csv adds mean_aa_len, median_aa_len, terminal_stop_rate, hard_cap_rate and aa_len_vs_k.png plot.

Secondary-Structure Analysis

- Heuristic propensities (unsupervised): scripts/ss_propensity.py
  - Inputs: --run_id (resolves primary_dna) or --dna path/to/cds.txt
  - Method: sliding-window averages of AA propensities (Chou–Fasman‑style tables), thresholds for helix/sheet calls.
  - Outputs: per-sequence counts/fractions/lengths; length histograms; merges median segment lengths into outputs/scores/<RUN_ID>/metrics.json when run_id is provided.
  - Caveat: correlation‑level only; not a structure predictor. Use to gauge plausibility and compare runs.

- Linear probe (supervised): scripts/probe_ss_linear.py
  - Inputs: NPZ with H (N,T,D) token embeddings, Y (N,T) labels in {0:C,1:H,2:E}, optional M (N,T) mask.
  - Outputs: metrics.json (accuracy, macro‑F1, AUROC when possible), confusion.png, and a saved probe.
  - Purpose: measure how linearly decodable H/E/C is from LM token embeddings when given labels (from PSIPRED/NetSurfP/DSSP, etc.).

Disorder Heuristics

- scripts/disorder_heuristics.py complements SS analysis by covering intrinsically disordered regions (IDRs).
  - Inputs: --run_id or --dna (CDS per line), optional --out_dir.
  - KPIs: charge–hydropathy (Uversky) point + class, net charge per residue (NCPR), kappa-like charge patterning proxy, fraction of disorder-promoting residues (E,D,K,R,Q,S,P,G), low-complexity segments via Shannon entropy windows.
  - Outputs: summary.csv and plots (CH-plane, LCR length and fraction histograms). Merges aggregate KPIs into outputs/scores/<RUN_ID>/metrics.json when run_id is provided.
  - Caveat: Heuristics, not predictors. For per-residue disorder probabilities, integrate a local predictor (IUPred2A, DISOPRED, etc.).

Modern propensity sources

- The bundled propensities reflect classic Chou–Fasman‑style rankings. For updated scales:
  - AAindex database (curated physico‑chemical/propensity indices).
  - Helix propensity scales (e.g., Pace & Scholtz) and strand propensities from large curated datasets.
  - GOR‑style methods use neighbor information (information theory) and outperform simple thresholds, but require model‑based scoring beyond this script’s scope.

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
Sequence Quality & Calibration

- scripts/seq_quality.py aggregates coding sanity and realism KPIs:
  - ORF integrity (start codon ATG/GTG/TTG; no internal stops; terminal stop present)
  - Length distribution (optionally Z‑scores vs reference), GC%
  - Codon usage divergence (KL/JS) vs reference usage from a provided table or reference CDS
  - CAI (Codon Adaptation Index) using supplied weights (or derived from usage by per‑AA normalization)
  - 3‑nt periodicity: FFT peak at 1/3 cycles/nt over a purine indicator signal
  - Diversity/novelty: k‑mer Jaccard and MinHash Jaccard vs a reference CDS set
  - Outputs per‑sequence summary.csv, histograms; merges headline KPIs into metrics.json when run_id is used.

- scripts/calibration_metrics.py computes ECE/Brier on a split using a checkpoint, ignoring PAD targets.

Theoretical notes

- ORF integrity and 3‑nt periodicity test coding‑like grammar (frame, codon structure).
- Codon usage and CAI reflect organism‑specific translational biases; KL/JS measures divergence from reference; CAI is a geometric mean of relative adaptiveness per codon.
- Diversity/novelty (k‑mer Jaccard/MinHash) ensure runs/generations aren’t memorizing; MinHash provides a fast approximation to set similarity.
- Calibration: ECE and Brier quantify the alignment between predicted probabilities and empirical outcomes (proper scoring); label smoothing and warmup schedules often improve ECE.
