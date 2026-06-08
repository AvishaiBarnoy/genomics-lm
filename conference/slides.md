---
marp: true
theme: default
paginate: true
backgroundColor: #fff
header: 'The Laptop Scientist | Genomics-LM'
footer: 'A PhD Journey into Genomic ML'
---

# 🧬 The Laptop Scientist
### Deciphering the "Language of Life" on a PhD Budget

---

# The Problem: "ML is for the 1%"
- I wanted to learn Machine Learning.
- Industry standard: A100 clusters, millions in cloud spend.
- My reality: A PhD stipend and a laptop.
- **The Question**: Can I do meaningful AI research with just what I have at home?

---

# The Subject: DNA as the Ultimate Database
- Every cell contains a "Database" (DNA).
- It stores:
  - **Executables**: Genes (instructions for proteins).
  - **Config files**: Regulatory regions.
  - **Metadata**: Epigenetics.
- **The Challenge**: How do we "query" this database without a SQL engine?

---

# Biology 101: The "Code"
- **DNA**: Strings of A, C, G, T.
- **Codons**: 3-letter "words" (e.g., `ATG`, `GCA`).
- **Translation**:
  - DNA (Database) -> mRNA (Buffer) -> Protein (Hardware).
- Each codon maps to an Amino Acid (The building blocks of cellular machinery).

---

# My Journey: Frugal AI
- **Constraint**: No GPUs. No Cloud.
- **Solution**: **Genomics-LM**.
- A compact "Codon-level" GPT model.
- Small enough to train while you drink your coffee.
- Large enough to "understand" the grammar of life.

---

# The Pipeline: Interrogating the Black Box
Since I couldn't afford *scale*, I focused on *depth*.
1. **Frequencies**: Does the model see the "punctuation"?
2. **Embeddings**: Can it group synonyms?
3. **Attention**: Where is the AI looking?
4. **Probing**: Can it pass a biology exam?

---

# Discovery #1: Autonomous Punctuation
- I never told the model what a "Start" or "Stop" codon was.
- **Result**: The model's internal representations identify these with **>95% accuracy**.
- It "discovered" the boundaries of the database entries on its own.

---

# Discovery #2: Physical Semantics
- Amino acids have physical properties: Charge, Size, Hydrophobicity.
- **Finding**: The model's embedding space naturally clusters these properties.
- It "learned" chemistry just by reading DNA strings.

---

# Scaling Up (The Hard Way)
- How do we get more from less?
- **Protein-Critic Bridge**: Using a "smart" supervisor to guide a "fast" model.
- **ReD Policy**: Preventing the model from getting stuck in repetitive loops.
- **Efficiency**: Optimizing for MacBook-friendly training.

---

# Conclusion: Empowerment
- ML is not just for big tech.
- By treating DNA as a language and a database, we can extract massive value from tiny models.
- If you're a scientist with a laptop: **You are an AI researcher.**

---

# Thank You!
- **Genomics-LM**: Interpretable, Frugal, Biological AI.
- [Your Name/Repo Link]
- **Questions?**
