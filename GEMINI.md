# Genomics-LM

## Project Overview
`genomics-lm` is a comprehensive toolkit for training, analyzing, and benchmarking language models on genomic data. It provides reproducible pipelines for two primary domains:
1.  **CodonLM**: A compact codon-level GPT-style language model for DNA/mRNA sequences.
2.  **ProteinLM**: A conditional transformer for protein sequences, supporting functional and topological conditioning.

The project emphasizes reproducibility, analysis (embeddings, attention, probing), and benchmarking against classical baselines.

**Roadmap Status:** Currently focusing on "Stage 1 – Toy Scale" (mechanics, interpretability pipeline) with plans for mid-scale adapters and large-backbone fine-tuning.

## Tech Stack
*   **Language:** Python
*   **Framework:** PyTorch (inferred from model architecture and checkpointing references)
*   **Environment:** Conda (managed via `environment.yml` or root `requirements.txt`)
*   **Configuration:** YAML
*   **Testing:** `pytest`

## Architecture
The codebase is structured into source modules and executable scripts:

### Source Code (`src/`)
*   **`codonlm/`**: Contains the TinyGPT model (`model_tiny_gpt.py`), trainer (`train_codon_lm.py`), and data utilities.
*   **`protein_lm/`**: Implements the `ProteinConditionalTransformer` and `ProteinClassifier`, along with a specialized tokenizer.
*   **`classifiers/`**: Downstream classification models (e.g., MLP, Probes).
*   **`eval/`**: Evaluation metrics and logic.

### Conceptual Architecture (Agents)
The project is conceptually divided into "agents" (modules) as defined in `docs/agents.md`:
1.  **Data Extraction**: `src/codonlm/extract_cds_from_genbank.py`
2.  **Tokenization**: `src/codonlm/tokenizer.py`
3.  **LM Training**: `src/codonlm/train_lm.py`
4.  **Classifier**: `src/codonlm/train_classifier.py`
5.  **Generation & Eval**: `scripts/eval_generation_prefix.py`
6.  **Motif Analysis**: (Planned)
7.  **Pipeline Orchestrator**: `pipeline.sh` (or similar)
8.  **Reporting**: `scripts/generate_report.py`

### Scripts (`scripts/`)
The `scripts/` directory is the main interface for users, containing tools for:
*   **Data Pipeline**: `pipeline_prepare.py`
*   **Training**: Wrappers and entry points (often invoked via `main.sh` or directly).
*   **Inference**: `infer_predict_next_codon.py`, `infer_generate_cds.py`.
*   **Analysis**: `analyze_attention.py`, `analyze_embeddings.py`, `compare_runs.py`.
*   **Benchmarking**: `eval_classifier.py`, `evaluate_test.py`, `sanity_kpis.py`.

### Configuration (`configs/`)
All experiments are driven by YAML configuration files, defining model hyperparameters, data paths, and training settings (e.g., `configs/tiny_mps.yaml`, `configs/protein_lm/small.yaml`).

## Workflows

### 1. Setup
Create the conda environment:
```bash
conda env create -f environment.yml
conda activate codonlm
```

### 2. Training
**CodonLM:**
Use the helper script for a default run:
```bash
./main.sh
```
Or run custom configurations via python scripts referenced in `main.sh`.

**ProteinLM:**
Train the language model:
```bash
python -m src.protein_lm.train_lm --config configs/protein_lm/small.yaml
```
Train the classifier:
```bash
python -m src.protein_lm.train_classifier --config configs/protein_lm/classifier_small.yaml
```

### 3. Analysis & Evaluation
Analyze a specific run (CodonLM):
```bash
./analysis.sh <RUN_ID> [optional_config]
```
Compare multiple runs:
```bash
python -m scripts.compare_runs
```
Query a trained model:
```bash
python -m scripts.query_model <RUN_ID> --mode next --dna ATGAAACCC --topk 5
```

## Development Conventions
*   **Run IDs**: Experiments are tracked via unique Run IDs. Outputs are stored in `runs/<RUN_ID>/checkpoints` and `runs/<RUN_ID>/scores`.
*   **Testing**: Run unit tests using `pytest`:
    ```bash
    pytest
    ```
    Key test files include `tests/test_models.py` and `tests/test_protein_models.py`.
*   **Style**: The project appears to follow standard Python formatting. Configuration files are central to ensuring reproducibility.
