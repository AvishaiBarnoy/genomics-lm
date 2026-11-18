# Genomics-LM

**A compact codon-level language model for bacterial DNA, with controlled evaluation of what it learns and generates.**

Genomics-LM trains a causal transformer to predict the next codon (a three-base DNA token). The repository includes genome extraction and tokenization, PyTorch training on limited hardware, sequence generation, downstream probes, baselines, and audit tools. The main experiments ran on a single Apple M2 Mac with 8 GB of unified memory.

The central question is whether next-codon prediction learns useful biological sequence patterns beyond simple codon statistics, and whether those patterns transfer to generation and other tasks. The answer so far depends on the task and the evaluation protocol.

## What we built

- **Data and training:** GenBank extraction, codon and hybrid tokenization, dataset manifests, a compact causal GPT, checkpointed training, and Apple Silicon MPS workflows.
- **Controlled evaluation:** frozen genome-level splits, held-out perplexity, train-fitted n-gram baselines, sequence controls, causal embedding extraction, and provenance checks.
- **Generation and analysis:** sampling, termination diagnostics, novelty audits, synonymous-codon controls, an external ESM-2 protein plausibility benchmark, and run dashboards.
- **Downstream experiments:** AMR classification, DNA-shape proxy regression, protein critics, and other exploratory probes with grouped splits and comparison models where available.

## Results so far

The corrected primary model is a 10-layer, 8-head, width-384 transformer trained on a frozen bacterial corpus with genome-level holdouts. Two locked training seeds beat a train-fitted, segment-aware trigram on the **same frozen test tokens**:

| Model | Test perplexity (lower is better) |
| --- | ---: |
| Trigram baseline | 42.037 |
| CodonLM, seed 1337 | 39.133 |
| CodonLM, seed 2027 | 39.492 |

See the [perplexity baselines](docs/PERPLEXITY_BASELINES.md) and [replication record](docs/benchmarks/corrected_lr15_replication.json) for token counts, checkpoint identities, and protocol details. These results establish an improvement in held-out next-codon prediction over this simple baseline; they do not establish biological function or a useful gene-level representation.

Other corrected evaluations give a more specific picture:

| Evaluation | Observed outcome |
| --- | --- |
| **Synonymous controls** | In a 100-gene validation pilot, the model preferred natural coding sequences over alternatives encoding the same proteins. It also preferred 80% of natural genes over within-gene codon shuffles, though that ordering effect was small. [Pilot report](docs/SYNONYMOUS_CONTROLS_PILOT_REPORT.md). |
| **Unrestricted generation** | Natural stops occurred in 35/50 and 28/50 samples for the two seeds. The remaining samples often reached the length cap, and completed sequences tended to be short. A top-k 5 setup produced no natural stops in a separate matched evaluation. [Generation results](docs/UNRESTRICTED_GENERATION_CONFIRMATION.md). |
| **AMR classification** | Corrected CodonLM embeddings beat a nucleotide 3-mer baseline on some class-aware metrics, but random transformer features performed better than the pretrained final-layer embeddings. Train-only selection of an earlier layer narrowed the gap without establishing a pretraining advantage. [AMR evaluation](docs/CORRECTED_AMR_EVALUATION.md). |
| **DNA-shape proxies** | Linear probes on corrected embeddings did not generalize across held-out genomes; local sequence and random-model controls performed better. The targets are computed from DNA sequence, not measured structures. [DNA-shape evaluation](docs/CORRECTED_DNA_SHAPE_EVALUATION.md). |
| **External protein critic** | In a 32-versus-32 pilot, a frozen ESM-2 8M model scored natural validation proteins above generated proteins. This is a plausibility comparison, not a functional assay. [Critic report](docs/ESM2_PROTEIN_CRITIC_REPORT.md). |

The corrected EC classification evaluation is **pending**: the available annotated records do not provide an EC test set outside the pretraining genomes. The controlled AMR test excludes exact pretraining CDS duplicates, but some protein clusters still overlap pretraining and are disclosed in the [downstream readiness report](docs/CORRECTED_DOWNSTREAM_EVALUATION_READINESS.md).

Earlier Stage 2.6 figures remain in the [historical benchmark table](conference/sota_benchmark_table.md). They predate mandatory global genome-aware splitting and corrected causal embedding extraction, so they should not be read as current validated performance claims.

## Getting started

```bash
git clone https://github.com/AvishaiBarnoy/genomics-lm.git
cd genomics-lm
conda env create -f environment.yml
conda activate codonlm
```

Read the [manual](docs/MANUAL.md) before preparing data or starting a substantive training run. The default command uses the small `configs/tiny_mps.yaml` workflow; it does **not** reproduce the corrected results above:

```bash
./main.sh
```

For an existing run, replace `<RUN_ID>` with its directory name under `runs/`:

```bash
./analysis.sh <RUN_ID>
python -m scripts.query_model <RUN_ID> --mode next --dna ATGAAACCC --topk 5
streamlit run scripts/web_dashboard.py
```

The corrected corpus, training configurations, and test commands are documented in the [dataset manifest guide](docs/DATASET_MANIFEST.md), [training configuration guide](docs/CORRECTED_PRIMARY_TRAINING_CONFIGS.md), and [intrinsic evaluation guide](docs/CORRECTED_PRIMARY_INTRINSIC_EVALUATION.md).

## Development checks

The required CPU checks target Python 3.11:

```bash
ruff check . --select E9,F63,F7,F82
pytest -q -m "not slow and not external and not mps" --cov=src --cov=scripts --cov-report=term
```

Slow, external-dependency, and Apple Silicon MPS tests run separately. The CPU suite checks software behavior, not scientific validity or MPS performance.

## Repository guide

| Location | Contents |
| --- | --- |
| `src/` | Models, data processing, training, and evaluation code |
| `scripts/` | Training, evaluation, audits, and analysis commands |
| `configs/` | Experiment configurations |
| `tests/` | Automated tests |
| `runs/` | Run checkpoints, metrics, and artifacts |
| `conference/` | Figures and historical benchmark summaries |
| `docs/` | Protocols, reports, and project history |

For the full history, see the [development log](docs/DEVELOPMENT_LOG.md), [manual](docs/MANUAL.md), and [research tracks](conductor/tracks.md).
