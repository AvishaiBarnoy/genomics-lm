# Technology Stack

## Core
- **Programming Language:** Python
- **ML Framework:** PyTorch (used for model architecture, training, and inference)
- **Environment:** Conda (environment defined in `environment.yml` or root `requirements.txt`)

## Data & Configuration
- **Configuration:** YAML (experiment parameters and model definitions)
- **Primary Data Formats:**
    - NPZ (compressed numpy arrays for tokenized datasets and artifacts)
    - JSON/CSV/TSV (metrics, logs, and summaries)
- **Biological Data Formats:**
    - Fasta (raw and processed sequences)
    - GenBank (.gbff) (annotated genomic records)

## Tools
- **Testing:** Pytest
- **Static Analysis:** Ruff (inferred from CI configuration)
- **Data Visualization & Tabulation:** Matplotlib, Scikit-learn (PCA), Pandas, Tabulate
- **Web Dashboard:** Streamlit
