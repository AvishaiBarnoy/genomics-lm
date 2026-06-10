# hayaData 2026 Submission Proposal: Genomics-LM

`genomics-lm` is an excellent fit for the **hayaData** conference. Because hayaData brings together data engineers, data scientists, researchers, and developers, you can frame this project from multiple angles depending on which track you want to target.

Below is a proposal detailing the **three primary submission categories** you can target, along with **four concrete talk concepts**, complete with titles, abstracts, key takeaways, and references to your code.

---

## 🛑 Summary of Tracks & Talk Options

| Category | Recommended Talk Title | Primary Code Elements Reference |
| :--- | :--- | :--- |
| **Data Science & ML** | *Probing the Hidden Mind of a Genomic GPT: How a 1D Language Model Learns 3D DNA Physics* | [probe_structural_awareness.py](file:///Users/User/github/genomics-lm/scripts/probe_structural_awareness.py), [analyze_dialects.py](file:///Users/User/github/genomics-lm/scripts/analyze_dialects.py) |
| **Data Science & ML** | *Beyond Text: Building a Generator-Critic Loop for Synthetic DNA & Protein Design* | [protein_critic_bridge.py](file:///Users/User/github/genomics-lm/scripts/protein_critic_bridge.py), [train_multi_task.py](file:///Users/User/github/genomics-lm/src/protein_lm/train_multi_task.py) |
| **Data Engineering** | *Operon-Aware Genomic Tapes: Building Big Data Pipelines for Biological Language Models* | [pipeline_prepare.py](file:///Users/User/github/genomics-lm/scripts/pipeline_prepare.py), [extract_cds_from_genbank.py](file:///Users/User/github/genomics-lm/src/codonlm/extract_cds_from_genbank.py) |
| **MLOps / Practical ML** | *GPT on a Shoestring Budget: Solving Context-Length Limits with Generator-Critic Re-feeding on a Laptop* | [benchmark_red.py](file:///Users/User/github/genomics-lm/scripts/benchmark_red.py), [protein_critic_bridge.py](file:///Users/User/github/genomics-lm/scripts/protein_critic_bridge.py) |

---

## 1. Category: Data Science & AI/ML
*This track is for ML developers, data scientists, and researchers. It focuses on models, architectures, interpretability, and new AI paradigms.*

### Option A: The Interpretability & Representation Probe Angle
This talk focuses on how language models trained on biological sequences learn structural chemistry without being explicitly taught physics.

> **Talk Title:** Probing the Hidden Mind of a Genomic GPT: How a 1D Language Model Learns 3D DNA Physics
>
> **Abstract:**
> What do autoregressive transformers actually learn when trained on biological sequences? In this session, we go under the hood of a causal genomic language model to analyze its latent spaces. 
> We show how a 1D codon-level GPT-style model learns to capture "bacterial dialects" (codon usage biases across taxonomic families) and, more importantly, implicitly maps 3D DNA physical properties. Using linear probing, we show how frozen hidden states correlate strongly with physical features like Minor Groove Width and Electrostatic Potential. 
> Attendees will learn practical methodologies for probing neural network representations, visualizing attention maps in sequence models, and interpreting what an LLM actually understands about physical structures.
>
> **Key Takeaways:**
> - How to build a custom linear probing framework for transformer hidden states.
> - Techniques for analyzing taxonomy/dialect biases in causal language models.
> - The connection between 1D next-token prediction and 3D physical stereochemistry.
>
> **Key Code Reference:**
> - Probing script: [scripts/probe_structural_awareness.py](file:///Users/User/github/genomics-lm/scripts/probe_structural_awareness.py)
> - Dialect analysis: [scripts/analyze_dialects.py](file:///Users/User/github/genomics-lm/scripts/analyze_dialects.py)

---

### Option B: The Generator-Critic Architecture Angle
This talk focuses on sequence design, solving the "termination problem," and combining generative autoregressive models with multi-task discriminators.

> **Talk Title:** Beyond Text: Building a Generator-Critic Loop for Synthetic DNA & Protein Design
>
> **Abstract:**
> While text LLMs generate paragraph-by-paragraph, biological generation faces a harder constraint: a single syntax error or structural collapse renders a protein entirely non-functional. 
> In this talk, we share how we built a two-stage generative pipeline for biological sequences. First, a causal GPT-style model generates candidate coding sequences. Second, a multi-task Protein Critic (an encoder transformer) evaluates the generated candidates for Pfam classification, EC functional class, and stability. We'll dive into the math and biology behind "Reset-and-Discard" (ReD) sampling, explaining how it overcomes the sublinear termination rates typical in autoregressive generators and shifts yield to a linear scale.
>
> **Key Takeaways:**
> - How to pair generative causal models (CodonLM) with downstream critics (ProteinLM) in a feedback loop.
> - How to train a multi-task protein classifier to predict functional properties.
> - Designing optimal sampling algorithms (Reset-and-Discard) for structural sequence constraints.
>
> **Key Code Reference:**
> - Critic bridge: [scripts/protein_critic_bridge.py](file:///Users/User/github/genomics-lm/scripts/protein_critic_bridge.py)
> - Multi-task training: [src/protein_lm/train_multi_task.py](file:///Users/User/github/genomics-lm/src/protein_lm/train_multi_task.py)
> - Reset-and-Discard benchmark: [scripts/benchmark_red.py](file:///Users/User/github/genomics-lm/scripts/benchmark_red.py)

---

## 2. Category: Data Engineering & Pipelines
*This track is for data engineers and system architects. It focuses on parsing massive raw datasets, custom tokenization, and pipeline efficiency.*

> **Talk Title:** Operon-Aware Genomic Tapes: Building Big Data Pipelines for Biological Language Models
>
> **Abstract:**
> Before training a language model on DNA, you have to transform raw, noisy biological records (like GenBank files) into structured training tokens. 
> This talk covers the data engineering pipeline behind Genomics-LM. We will detail how to parse CDS regions, handle taxonomies, implement custom codon-level tokenization, and generate sliding-window "Genomic Tapes." We'll focus on how data selection solves model behavior: how hard-mining "Anchored Operon Bridges" (sequences centered exactly on gene boundaries) corrected a 0.0% natural termination rate, proving that target data selection can be more effective than model architecture changes.
>
> **Key Takeaways:**
> - Designing specialized data preparation pipelines for sequence tokenization.
> - Handling boundary conditions and taxonomic skew during dataset construction.
> - Data-centric AI: Hard-mining transition boundaries to change downstream generation behavior.
>
> **Key Code Reference:**
> - Data preparation: [scripts/pipeline_prepare.py](file:///Users/User/github/genomics-lm/scripts/pipeline_prepare.py)
> - GenBank parsing: [src/codonlm/extract_cds_from_genbank.py](file:///Users/User/github/genomics-lm/src/codonlm/extract_cds_from_genbank.py)

---

## 3. Category: MLOps / Practical ML
*This track focuses on model training efficiency, deployment, and getting high performance out of limited hardware.*

> **Talk Title:** GPT on a Shoestring Budget: Solving Context-Length Limits with Generator-Critic Re-feeding on a Laptop
>
> **Abstract:**
> Training language models on long genomic sequences is exceptionally hard under local hardware constraints. Autoregressive causal models need massive context windows (block sizes) to learn sequence termination and gene boundaries, but this leads to quadratic memory growth and out-of-memory (OOM) errors on consumer laptops.
> 
> In this case study, we demonstrate how we trained Genomics-LM on an Apple M2 8GB MacBook by solving the context-limit problem architecturally. Instead of scaling up a single massive model, we split the task: we trained a lightweight causal DNA-level generator (CodonLM) and a separate multi-task Protein Critic (ProteinLM) on the biological "next step" (the translated amino acid sequence). By implementing a Generator-Critic loop with a Reset-and-Discard (ReD) re-feeding policy, we generated structurally valid, naturally terminating sequences while keeping GPU RAM under 8GB.
>
> **Key Takeaways:**
> - How to solve long-context memory limits by decoupling generation from validation/criticism.
> - Implementation of a next-step biological re-feeding mechanism (DNA translation -> Protein validation).
> - Practical RAM-saving deep learning tips: Scaled Dot Product Attention (SDPA), gradient accumulation, and state caching.
>
> **Key Code Reference:**
> - Critic bridge: [scripts/protein_critic_bridge.py](file:///Users/User/github/genomics-lm/scripts/protein_critic_bridge.py)
> - Re-feeding & sampling policy: [scripts/benchmark_red.py](file:///Users/User/github/genomics-lm/scripts/benchmark_red.py)
> - Local playground: [scripts/web_dashboard.py](file:///Users/User/github/genomics-lm/scripts/web_dashboard.py)
