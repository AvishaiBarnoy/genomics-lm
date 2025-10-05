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

Running the training script is easy with:

```
chmod +x pipeline.sh
./pipeline.sh
```

## How to run the 6-step pipeline

```bash
RUN_ID=2025-09-30_tiny_2L4H_d128_e5
python -m scripts.collect_artifacts_yaml $RUN_ID path/to/tiny_mps.yaml
python -m scripts.analyze_frequencies   $RUN_ID
python -m scripts.analyze_embeddings    $RUN_ID
python -m scripts.analyze_attention     $RUN_ID
python -m scripts.probe_next_token      $RUN_ID
python -m scripts.analyze_saliency      $RUN_ID
python -m scripts.probe_linear          $RUN_ID
python -m scripts.summarize_one_cds     $RUN_ID  # optional
python -m scripts.compare_runs $RUN_ID <other_run_ids...>
```

