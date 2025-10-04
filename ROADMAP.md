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

## The 6‑Step Interpretability Pipeline (run after **every** training)

*Purpose: Watch the “language” your model learns evolve as capacity scales.*

> All steps consume `runs/<run_id>/artifacts.npz` (plus optional labels) and write to `runs/<run_id>/{charts,tables}`.

**Step 1 — Frequencies & Positional Punctuation**
Script: `analyze_frequencies.py`
Outputs: `tables/frequencies.csv`, `charts/top20_freq.png`, `charts/first_position_counts.png`
Focus: ATG as “first word”, stop codons as “periods”, position‑wise enrichments.

**Step 2 — Embedding Semantics**
Script: `analyze_embeddings.py`
Outputs: `charts/emb_pca.png`, `tables/nearest_neighbors.csv`, `tables/embed_quality.txt`
Focus: clustering of synonymous codons; separation of stops/specials; cosine‑NN “thesaurus.”

**Step 3 — Attention Specialization**
Script: `analyze_attention.py`
Outputs: `charts/attn_L{l}_H{h}_{t0}-{t1}.png`
Focus: heads that track start/stop, frame maintenance, or motif boundaries.

**Step 4 — Contextual Next‑Token Probes**
Script: `probe_next_token.py`
Outputs: `tables/next_token_tests.csv`
Focus: conditional distributions after prefixes (`ATG`, `ATG‑AAA`, `TAA`, …) as grammar tests.

**Step 5 — Saliency / Attribution**
Script: `analyze_saliency.py`
Outputs: `tables/saliency.csv`
Focus: which tokens drive predictions; spikes at conserved motifs, `ATG`, in‑frame stops.

**Step 6 — Biology‑Aware Linear Probes**
Script: `probe_linear.py`
Inputs: `probe_labels.csv` (token→AA, polarity, hydropathy, is\_stop, is\_start).
Outputs: `tables/probe_results.csv`
Focus: how well embeddings linearly decode AA identity/properties; improves with scale.

**(Optional) TSV Summaries**
Script: `summarize_one_cds.py`
Outputs: `tables/one_cds__summary.csv` (+ per‑metric CSVs if present).

---

## Stage 2 – Mid‑Scale (Pretrained models + adapters)

*Goal: Add functional prediction and motif‑conditioned generation while keeping the same 6‑step lens.*

**Projects**

1. **Protein Classifier with LoRA** (ESM‑2 35M / ProtT5 60M).

* Fine‑tune LoRA on labelled data (EC classes, AMR genes).
* Evaluate AUROC/F1; run Steps 2, 5, 6 on the adapter’s embedding space.

2. **Function‑Conditioned Generator**

* Add control tokens (e.g., `<EC_3>`).
* Validate motif preservation in outputs; run Step 4 on controlled prefixes.

3. **Motif Co‑occurrence Graphs**

* Build motif graphs from hidden‑state clusters; relate to function labels; inspect with Steps 2–3.

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

