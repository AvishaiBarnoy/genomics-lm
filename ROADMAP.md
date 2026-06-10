# 🧬 Protein/Genomics LM Roadmap

---

## Stage 1 – Toy Scale (MacBook‑friendly)

*Goal: Learn mechanics — tokenization, training, **and** a repeatable 6‑step interpretability pipeline.*

**Minimal model**: 2 layers · 4 heads · d=128 · 5 epochs (YAML‑driven).
**Core outputs per run**: `runs/<run_id>/config.yaml`, `weights.pt`, `artifacts.npz`, `itos.txt`, optional `motif_clusters.npz`, `one_cds__best.tsv`.

### Projects

1. **Codon Language Model**

* Train on a handful of bacterial genomes (\~200k–1M params).
* Evaluate base LM metrics (loss, perplexity).
* Produce artifacts with `collect_artifacts_yaml.py`.

2. **Mutation Maps (gene‑level probing)**

* For a gene (e.g., *lacZ*), compute Δ log‑likelihood for single‑codon substitutions.
* Compare "hot" positions vs conservation and known motifs.

3. **Motif Mining**

* Extract hidden‑state embeddings (from artifacts) and cluster subsequences.
* Validate candidate motifs against known protein family motifs or positional conservation.

---

## The 6‑Step Interpretability Pipeline (summary)

High‑level goals per step (full details in MANUAL.md):

- Step 1 — Frequencies: codon usage, first‑position patterns, stops/starts as punctuation.
- Step 2 — Embeddings: PCA/NN semantics; synonymous clusters; quality scores.
- Step 3 — Attention: heads specializing to starts/stops, frames, motif boundaries.
- Step 4 — Next‑token probes: conditional distributions after biological prefixes.
- Step 5 — Saliency: which tokens/regions drive predictions; motif spikes.
- Step 6 — Linear probes: decode AA identity/properties from embeddings.

All steps read `runs/<run_id>/artifacts.npz` (plus optional labels) and write charts/tables under `runs/<run_id>/`.

---

## Stage 2 – Mid‑Scale (Pretrained models + adapters)

*Goal: Add functional prediction and motif‑conditioned generation while keeping the same 6‑step lens.*

**Status: IN PROGRESS (Groundwork Complete)**
- [x] Diversified Bacterial Dataset (Gram-pos, High-GC, Entero)
- [x] Transfer Learning Pipeline (Fine-tuning support)
- [x] **Hierarchical Supervisor Model**: Use ProteinLM to 'critic' CodonLM outputs.
  - [x] **Protein-Critic Bridge**: DNA -> ProteinLM scoring script (`scripts/protein_critic_bridge.py`).
  - [x] **Multi-Task Classifiers**: Multi-head model for Family, Stability, and Function.
- [x] **Inference Policy Optimization (ReD)**: Implement Reset-and-Discard to overcome the sublinear coverage trap and the 0% termination barrier.
- [x] **Termination Motif Diagnostics**: 
  - [x] Created sampling auditor (`scripts/check_termination_motifs.py`).
  - [x] Created causal in-silico perturbation (`scripts/test_perturbation_motifs.py`).
  - [x] Created post-STOP UTR generation check (`scripts/test_utr_generation.py`).
  - [x] Verified CodonLM currently lacks structural terminator generation (see `docs/termination_motifs_analysis.md`).
- [ ] **Anchored Operon Bridges**: Train specifically on gene-gene boundaries to master operon logic. (In Progress)
- [ ] **Inference Benchmark**: Increase termination rate beyond 20% with Operon training. (In Progress)

---

## Stage 2.6 – The Multi-Scale Biophysical Architecture (Next Milestone)

*Goal: Overcome the 9x attention/memory scaling trap of nucleotide-level modeling while capturing fine-grained UTR physics.*

- [ ] **Dual-Track Late Fusion (Structural Compass)**:
  - Train a lightweight nucleotide encoder (e.g. CNN/local Transformer) on 60 bp sliding windows.
  - Inject physical DNA shape/energy representations into the main codon-level generator to guide sequence generation.
- [ ] **Hybrid Tokenization (Variable-Scale Tape)**:
  - Build a multi-scale tokenizer with a unified vocabulary (64 codons + 4 nucleotides).
  - Tokenize coding regions (CDS) as codons ($3\times$ compression) and intergenic/UTR regions as single nucleotides (1 bp resolution).
- [ ] **Energy-Based mRNA Optimizer (EBM)**:
  - Integrate a bidirectional Energy-Based Model at the nucleotide level to optimize synonymous codon sequences for global stability ($\Delta G$).

**Projects**

1. **Protein Classifier with LoRA** (ESM‑2 35M / ProtT5 60M).

* Fine‑tune LoRA on labelled data (EC classes, AMR genes).
* Evaluate AUROC/F1; run Steps 2, 5, 6 on the adapter’s embedding space.

2. **Function‑Conditioned Generator**

* Add control tokens (e.g., `<EC_3>`).
* Validate motif preservation in outputs; run Step 4 on controlled prefixes.

3. **Motif Co‑occurrence Graphs**

* Build motif graphs from hidden‑state clusters; relate to function labels; inspect with Steps 2–3.

**Maintenance & Usability**

* [x] **Data Organization Consolidation**: Consolidate `outputs/` checkpoints/scores and `runs/` diagnostics under a unified run directory layout (`runs/<run_id>/`). Write unit verification tests and update documentation.
* [x] **Model Querying UI**: Created a visual "Model Playground" tab in the Streamlit web dashboard to automate next-codon, sequence generation, and protein classification queries.
* [x] **Model Querying UI Upgrades**: Integrated Reset-and-Discard (ReD) visualizer logs, interactive 3D DNAshape aligned profile plots, head-level attention heatmaps (via selective SDPA bypass), and synonymous mutation alignment comparison views.
* [x] **Live Training Progress Monitor**: Built a status tab that reads curves.csv and metrics.json dynamically during active training, plotting loss curves and rates.
* [x] **Remote Bioinformatics SQLite Caching**: Implemented EBI/NCBI BLAST API client fallbacks integrated with local SQLite caching for rate-limited auto-annotations.
* [x] **Stage 2.5: Genomic Tapes & Anchored Operons (Completed)**:
  * *Phase 1: Tape Extraction Logic* - sliding genomic windows over chromosomes.
  * *Phase 2: Tokenization & Dataset Packing* - single-mode packed datasets.
  * *Phase 3: Master Model Fine-tuning* - 6L4H 512-block context model resolving natural termination.
* [x] **SOTA Benchmarking & Hardware Profiling (Completed)**:
  * *Phase 1: Benchmark Data Acquisition* - mock/synthetic loaders for protein/rRNA DMS and gene essentiality.
  * *Phase 2: Zero-Shot Mutation Scoring Pipeline* - rank correlation metrics.
  * *Phase 3: Gene Essentiality Classification* - linear probing on sequence embeddings.
  * *Phase 4: Comparative Reports* - compute efficiency density reporting vs. Evo 1 and GenSLM.
* [ ] **Hybrid DNA-Protein Critic Benchmark**: Integrate the Multi-Task Critic as a bidirectional re-feeding filter (Phase 5 of SOTA Benchmarking).
* [ ] **Protein Latent Energy-Based Model**: Implement latent Langevin dynamics and NCE training for stability design.
* [ ] **Multi-Frame Overlapping Gene Modeling**: Sum frame-shift positional context embeddings to predict overprinted genes in viral/bacterial genomes.

**Hardware**: 16 GB MacBook with quantization+LoRA.
**Outcome**: Strong lightweight classifiers, interpretable embeddings, motif‑aware generation.

---

## Stage 3 – Stretch Goals (Cloud or Cluster)

*Goal: Approach genuine design tasks with the interpretability pipeline as a safety net.*

**Projects**

1. **Large‑Backbone Fine‑Tuning** (150M–650M ESM/ProGen).

* Keep Steps 1–6 to diagnose regressions and capacity‑driven gains.

2. **Motif‑Guided Design**

* Compose motifs A+B with learned linkers; validate with AF/ColabFold; iterate.

3. **Toward De‑novo Design**

* Combine LM generation with structure/energy scores; use RL or preference optimization.
* Use Steps 2–6 to ensure interpretability and avoid mode‑collapse.

4. **Multi-Scale Genomic Modelling**

* Implement nucleotide-level and parallel-frame (Frame 0, +1, +2) processing.
* Target: Model overlapping/overprinted genes in high-density prokaryotic/viral genomes.

5. **Hardware Scaling Ablation (Speculative)**

* Test the jump from 8GB/M2 to 16GB/M4.
* Scaling Targets: 2048 block size, 16 layers, 100+ genome diversity.
* Goal: Quantify the 'Capability Leap' provided by unified memory expansion.

**Hardware**: Cloud GPUs (A100/H100) or university cluster.
**Outcome**: First de‑novo functional design experiments.

---

## 🚀 Suggested Path (Learning Ladder)

1. **Start Small** — train minimal codon LM; run Steps **1–6**.
2. **Probe** — mutation heatmaps + motif clustering; relate to Steps **2,5**.
3. **Scale Up** — widen/deepen; compare runs with `compare_runs.py` (see below).
4. **Upgrade** — adapter‑tune ESM‑2 35M; interpret with Steps **2–6**.
5. **Stretch** — move to cloud; motif‑guided design with continual interpretability checks.

---

## 📊 Comparing Runs ("growth" in language understanding)

Script: `compare_runs.py`
Inputs per run: `metrics` (val perplexity), `tables/embed_quality.txt` (silhouette, PCA var), `tables/probe_results.csv` (AA/polarity/hydropathy/start/stop).
Output: `runs/_summary/summary.csv` with columns `[run_id, val_ppl, silhouette, probe_aa, probe_class_pol, probe_class_hydro, probe_is_stop, probe_is_start, …]`.

**What should improve with capacity/epochs?**

* ↓ Validation perplexity.
* ↑ Embedding silhouette by AA identity.
* ↑ Linear‑probe accuracies (AA, polarity, hydropathy).
* Crisper attention specializations; stronger context sensitivity in Step 4; clearer saliency structure in Step 5.

---

## ✅ Summary

On laptops you can **train toy codon LMs and run a principled 6‑step interpretability suite** after every run. The same pipeline scales to adapters and large models, giving you a consistent, biology‑aware view of how the model’s “language” grows with capacity.
4. **Remote Bioinformatics Integrations (optional)**

* Add optional remote BLAST/EBI requests with caching and rate‑limits to annotate generated sequences; keep disabled by default (local‑first philosophy).
* Roadmap item only; not implemented yet.
