# 🧬 Protein/Genomics LM Roadmap

---

## Stage 1 – Toy Scale (MacBook-friendly)
*Goal: Learn mechanics — tokenization, training, evaluation, motif analysis.*

**Projects**
1. **Codon Language Model**
   - Train a ~200k–1M parameter Transformer on a handful of bacterial genomes.
   - Capabilities: codon usage patterns, start/stop codon prediction, mutation scoring.

2. **Mutation Maps**
   - For a gene (e.g., lacZ), substitute each codon systematically.
   - Compute Δ log-likelihood → highlight critical vs. flexible residues.
   - Compare against known conserved motifs.
   - Train on diverse genomes (organisms/taxonomy mix).
   - AA-frozen shuffle: randomize synonymous codons (preserve AA seq) → retrain → compare ΔlogP.
   - Train a small AA-level LM and compare per-position sensitivities to codon-level.

3. **Motif Mining**
   - Extract hidden-state embeddings.
   - Cluster subsequences → candidate motifs.
   - Validate against known protein family motifs.

**Hardware:** MacBook M2 Pro (8 GB) or M4 Air (16 GB).  
**Outcome:** Understand LM training end-to-end.

---

## Stage 2 – Mid-Scale (Pretrained models + adapters)
*Goal: Add functional prediction, motif-conditioned generation.*

**Projects**
1. **Protein Classifier with LoRA**
   - Use pretrained ESM-2 (35M) or ProtT5 (60M).
   - Fine-tune a LoRA head on labelled data (enzyme EC classes, antibiotic resistance genes).
   - Evaluate accuracy (AUROC, F1).

2. **Function-Conditioned Generator**
   - Add control tokens (e.g. `<EC_3>`).
   - Generate sequences conditioned on function label.
   - Validate motifs are preserved in outputs.

3. **Motif Co-occurrence Graphs**
   - Use embeddings to map which motifs appear together across classes.
   - Build graph/network visualization.
   - Hypothesize functional dependencies.

**Hardware:** 16 GB MacBook with quantization + LoRA.  
**Outcome:** Strong classifier + biased generation.

---

## Stage 3 – Stretch Goals (Cloud or Cluster Compute)
*Goal: Approach genuine design tasks.*

**Projects**
1. **Larger Protein LM Fine-Tuning**
   - Use 150M–650M param ESM or ProGen backbones.
   - Fine-tune adapters on large labelled datasets (>100k seqs).

2. **Motif-Guided Design**
   - Take motifs A + B, ask LM to fill linker.
   - Validate with AlphaFold / ColabFold.
   - Iteratively filter designs by predicted structure/stability.

3. **Exploration toward De-novo Design**
   - Combine LM generation with structure scoring (AlphaFold confidence, Rosetta energy).
   - Use reinforcement learning or preference optimization to bias toward stability or binding.

**Hardware:** Cloud GPUs (A100/H100) or university cluster.  
**Outcome:** First experiments in *de-novo functional design*.

---

## 🚀 Suggested Path (Learning Ladder)

1. **Start Small**  
   Train a 500k-param codon LM on a single genome (Toy).

2. **Explore**  
   Run mutation heatmaps + motif clustering (Toy).

3. **Upgrade**  
   Download ESM-2-35M, fine-tune a LoRA classifier on enzyme classes (Mid-scale).

4. **Combine**  
   Generate sequences conditioned on class, check motifs preserved (Mid-scale).

5. **Stretch**  
   When ready, port pipeline to cloud → motif-guided design + AlphaFold screening (Stretch).

---

✅ **Summary**  
On MacBooks you can **train toy codon LMs, build strong classifiers with adapters, and explore motif logic**, then later scale those same skills into **function-conditioned design and de-novo experiments** with larger compute.

