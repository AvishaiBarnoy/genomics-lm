# Agents Overview

This document describes the logical “agents” (modular components) in **Genomics-LM**.  
Each agent handles a specific part of the DNA/codon–language modelling pipeline: from data ingestion to model training to downstream evaluation and analysis.

---

## 1. Data Extraction Agent  
**Script(s):** `src/codonlm/extract_cds_from_genbank.py` (and any associated helper modules)  
**Input:** GenBank (.gbff) files (annotated genomes)  
**Output:** Processed codon-level (or nucleotide/codon) sequence files (e.g., `.txt` or `.json`) in `data/processed/`  
**Purpose:** Converts annotated genome records into a consistent tokenisable dataset (codons, reading frames, filtering out pseudogenes, etc.) that serves as input for language modelling.  
**Notes:**  
- Handles multi-CDS records  
- Maintains reading‐frame integrity  
- Option to skip non-coding or ambiguous records  
- Downstream agents expect a uniform tokenisation vocabulary

---

## 2. Tokenisation & Vocabulary Agent  
**Script(s):** (e.g., `src/codonlm/tokenizer.py` or equivalent)  
**Input:** Preprocessed codon sequences from Extraction Agent  
**Output:** Tokenised datasets, vocabulary files (e.g., `vocab.json`, `merges.txt` or similar) stored under `data/vocab/`  
**Purpose:** Defines and builds the tokenisation scheme (codon tokens, special tokens like <PAD>, <EOS>, etc.), maps sequences to token IDs, prepares input for the model.  
**Notes:**  
- Choose between codon-level vs nt-level tokens  
- Document the vocabulary size, special tokens, handling of unknowns  
- Ensure reproducibility of tokenisation between training and inference

---

## 3. Language Model Training Agent  
**Script(s):** `train.py`, `src/codonlm/train_lm.py` (or similar)  
**Input:** Tokenised datasets (from Tokenisation Agent)  
**Output:** Trained model checkpoint(s) under `outputs/checkpoints/`, training logs (loss, perplexity), configs under `configs/`  
**Purpose:** Learns generative language modelling of codon (or nucleotide) sequences — e.g., causal LM, transformer-based architecture — enabling sequence generation, embedding extraction, etc.  
**Parameters:** Controlled via YAML or CLI flags (e.g., `block_size`, `n_layer`, `n_embd`, dropout, learning rate schedule, gradient accumulation).  
**Notes:**  
- Include training metrics and monitoring (e.g., via TensorBoard)  
- Consider reproducibility: set random seeds, log versions  
- Might include early stopping, checkpointing logic  
- Should integrate with downstream classifier or generation evaluation agents

---

## 4. Classifier / Downstream Task Agent  
**Script(s):** `src/codonlm/train_classifier.py`, `src/codonlm/evaluate_classifier.py`  
**Input:** Embeddings or representations from the LM Agent (or tokenised sequences directly), labelled data (e.g., protein family, function, organism)  
**Output:** Trained classifier model, evaluation metrics (accuracy, ROC‐AUC, confusion matrix), saved under `outputs/classifier/`  
**Purpose:** Applies the learned representations (or model) for classification / prediction tasks (for example: protein family classification, functional category prediction).  
**Notes:**  
- Define clearly what task(s) this agent covers  
- Clearly articulate the input data format and expected labels  
- Document evaluation protocol (train/dev/test splits, metrics)

---

## 5. Generation & Evaluation Agent  
**Script(s):** `scripts/eval_generation_prefix.py`, `tests/test_eval_generation_prefix.py`  
**Input:** Generated sequences from the LM training agent, ground-truth sequences (optional)  
**Output:** Evaluation results (CSV, plots) under `outputs/scores/`, with metrics such as stop‐codon behavior, frame integrity, motif recovery, sequence similarity.  
**Purpose:** Quantifies how well the generative LM performs biologically (not just in terms of loss/perplexity) — checks for biologically meaningful constraints.  
**Notes:**  
- Define set of biological criteria for validity (reading frame preserved, codon usage acceptable, stop codon placement, motif preservation)  
- Automate reporting of generation batches for reproducibility  
- Optionally integrate visualisation of sequence statistics

---

## 6. Motif / Functional Analysis Agent (Planned)  
**Script(s):** TBD (`src/codonlm/motif_analysis.py`, etc.)  
**Input:** Generated sequences, embeddings from LM, reference databases (e.g., Pfam, CDD)  
**Output:** Analysis reports — comparison of generated motifs/subsequences with known motifs, enrichment statistics, sequence‐function correlation plots.  
**Purpose:** Bridges generative modelling with downstream biological interpretation — e.g., do the generated sequences recapitulate known structural/functional motifs?  
**Status:** Planned for Stage 3 of the roadmap.  
**Notes:**  
- Might integrate BLAST, motif‐search tools, HMMER  
- Should produce a summary report (Markdown or HTML) linking sequence generation to functional hypothesis  
- Optionally integrate into pipeline orchestrator

---

## 7. Pipeline Orchestrator Agent  
**Script(s):** `pipeline.sh` (or a Python CLI script, e.g., `codonlm/cli.py`)  
**Purpose:** Coordinates all stages: from raw genome files → extraction → tokenisation → LM training → classification/regression tasks → evaluation → analysis.  
**Notes:**  
- Should allow modular invocation of each agent (e.g., `make data-extract`, `make train-lm`, `make evaluate`)  
- Logging and checkpointing across stages are important  
- Consider a config file (`pipeline.yml`) to toggle which agents to run, parameters, paths  
- For reproducibility, record software environment (e.g., `requirements.txt`, `conda.yaml`, Dockerfile if used)

---

## 8. Reporting / Monitoring Agent (Optional)  
**Script(s):** `scripts/generate_report.py`, `scripts/monitor_training.py`  
**Purpose:** Aggregates logs, checkpoints, evaluation metrics into summary dashboards or Markdown/HTML reports for experiments.  
**Notes:**  
- Useful for experiment tracking (learning curves, hyperparameter sweeps)  
- Could integrate with `wandb`, `mlflow`, or similar  
- Should allow browsing results of past runs via `outputs/reports/`

---

### 🧾 References  
- See the project’s `ROADMAP.md` for high-level stages of the work.  
- The `configs/` directory contains YAML files governing each agent’s parameters.  
- The `outputs/` directory holds results by agent category.  
- Contributors should update this document when adding new agents or modifying existing ones.

---

### 📌 How to Update This File  
Whenever a new logical component or “agent” is added to the pipeline (for example: data-augmentation agent, variant-effect prediction agent, etc.), please:  
1. Append a new section in this `agents.md` describing script, input, output, purpose.  
2. Add its config location and output directory to this overview.  
3. Ensure the README or docs include links/reference to where this agent is invoked.

---

_End of document._

