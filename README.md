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

Running the training scrpt is easy with:

```
chmod +x pipeline.sh
./pipeline.sh
```

