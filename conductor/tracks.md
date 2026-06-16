# Project Tracks

This file tracks all major tracks for the project. Each track has its own detailed plan in its respective folder.

---

- [x] **Track: Motif Mining & Cluster Analysis**
*Link: [./tracks/motif_mining_20260210/](./tracks/motif_mining_20260210/)*

- [x] **Track: Stage 2 Data Scaling & Transfer Learning**
*Link: [./tracks/stage2_scaling_20260603/](./tracks/stage2_scaling_20260603/)*

- [x] **Track: Genomic Tape Extraction**
*Link: [./tracks/genomic_tape_20260604/](./tracks/genomic_tape_20260604/)*

- [x] **Track: ReD Sampling for Inference Optimization**
*Link: [./tracks/red_sampling_20260607/](./tracks/red_sampling_20260607/)*

- [x] **Track: Shabbat 26-Hour Automated Workflow**
*Link: [./tracks/shabbat_workflow_20260605/](./tracks/shabbat_workflow_20260605/)*

- [x] **Track: Data Organization Consolidation**
*Link: [./tracks/data_organization_20260608/](./tracks/data_organization_20260608/)*

- [ ] **Track: NoProp Algorithm Integration**
*Link: [./tracks/noprop_integration_20260608/](./tracks/noprop_integration_20260608/)*
*Summary: Prototype architecture, trainer, config, and unit tests exist. No substantive training or memory-scaling validation has been completed; keep open until a real NoProp run is evaluated against standard backprop.*

- [x] **Track: Model Querying Streamlit UI**
*Link: [./tracks/querying_ui_20260608/](./tracks/querying_ui_20260608/)*

- [x] **Track: SOTA Benchmarking & Hardware Profiling**
*Link: [./tracks/sota_benchmarking_20260609/](./tracks/sota_benchmarking_20260609/)*

- [ ] **Track: hayaData 2026 Submission Preparation**
*Link: [./tracks/hayadata_submission_20260609/](./tracks/hayadata_submission_20260609/)*

- [ ] **Track: Multi-Scale Biophysical Architecture (Stage 2.6)**
*Link: [./tracks/multiscale_biophysics_20260609/](./tracks/multiscale_biophysics_20260609/)*

- [ ] **Track: Protein Latent Energy-Based Model (Stage 2.6)**
*Link: [./tracks/protein_ebm_20260610/](./tracks/protein_ebm_20260610/)*

- [ ] **Track: Hybrid DNA-Protein Critic Benchmark (Stage 2.6)**
*Link: [./tracks/hybrid_critic_20260610/](./tracks/hybrid_critic_20260610/)*

- [ ] **Track: Multi-Frame Overlapping Gene Modeling (Stage 3)**
*Link: [./tracks/multi_frame_overlapping_20260610/](./tracks/multi_frame_overlapping_20260610/)*

- [ ] **Track: Progressive High-Capacity Scaling Ladder (Stage 2.7)**
*Link: [./tracks/progressive_scaling_20260610/](./tracks/progressive_scaling_20260610/)*
*Summary: d384 ladder completed through 4L2H -> 6L4H -> 10L8H, and Stage 2.6 10L8H_d384 became the best current CodonLM. Track remains open for d384/d512 comparison closeout and the missing cross-width checkpoint expansion utility.*

- [x] **Track: Remote Bioinformatics Integrations (Maintenance)**
*Link: [./tracks/remote_bioinformatics_20260610/](./tracks/remote_bioinformatics_20260610/)*

- [x] **Track: Interactive UI Playgrounds & Live Monitor Upgrades (Maintenance)**
*Link: [./tracks/ui_improvements_20260610/](./tracks/ui_improvements_20260610/)*

- [x] **Track: Termination Fix & Dynamic Context Windows**
*Link: [./tracks/termination_fix_20260611/](./tracks/termination_fix_20260611/)*

- [x] **Track: Regression Probing for DNA Shape Decoding**
*Link: [./tracks/regression_probes_20260614/](./tracks/regression_probes_20260614/)*

- [x] **Track: Large Data-Scaling for Taxonomic Diversity**
*Link: [./tracks/large_data_scaling_20260614/](./tracks/large_data_scaling_20260614/)*

- [x] **Track: Training Speed & Memory Optimization**
*Link: [./tracks/training_speed_optimization_20260615/](./tracks/training_speed_optimization_20260615/)*
*Summary: All 5 phases implemented (GQA, mmap, BucketBatchSampler, CUDA device priority, SDPA path). Benchmark shows MPS batch=4 is dispatch-bound — optimizations benefit RAM/params, not throughput at this scale. CUDA batch≥32 expected to show ≥1.5× speedup.*

- [x] **Track: AMR Classification Probe (Conference)**
*Link: [./tracks/amr_classification_20260615/](./tracks/amr_classification_20260615/)*

- [x] **Track: EC & AMR Downstream Evaluation (Conference)**
*Summary: Completed EC Level-1 probe (AUROC=0.703), AMR probe (AUROC=0.967), k-mer baselines, UMAP+attention figures, SOTA table consolidation.*

- [x] **Track: Generative Design Loop**
*Link: [./tracks/generative_design_loop_20260615/](./tracks/generative_design_loop_20260615/)*
*Summary: ReD sampling + MultiTask ProteinCritic scoring + ESMFold API. Closes the generation→structure evaluation loop. 50/50 sequences terminated, pairwise identity 9.2%, ESMFold pLDDT ≈ 0.4–0.6 (novel/disordered — improvement direction: critic-guided ReD).*

- [x] **Track: Structured Protein Generation**
*Link: [./tracks/structured_generation_20260616/](./tracks/structured_generation_20260616/)*
*Summary: Closed as an experimental finding. Critic-guided ReD, family filtering, annealing, top-p sampling, and a structured-prefix harness were implemented; critic stability improved (+13.6%), but ESMFold pLDDT did not. Report: [./tracks/structured_generation_20260616/report.md](./tracks/structured_generation_20260616/report.md).*

- [ ] **Track: PDB-Filtered Structural Fine-Tuning**
*Link: [./tracks/pdb_structural_finetuning_20260616/](./tracks/pdb_structural_finetuning_20260616/)*
*Summary: Operational structural-training-signal path. Metadata enrichment and exact translated-CDS-to-UniProt structure filtering are implemented; 884/44,953 CDS were selected and the full 3-epoch Stage 3 run improved structure-subset validation loss/perplexity (4.088 → 4.068; ppl 59.75 → 58.45). Next required step is a matched ESMFold comparison to determine whether this improves pLDDT.*

- [ ] **Track: Structural-Aware ProteinCritic**
*Link: [./tracks/structural_aware_protein_critic_20260616/](./tracks/structural_aware_protein_critic_20260616/)*
*Summary: Add protein-type and foldability labels to ProteinCritic, including soluble, membrane, signal/secreted, disordered/low-complexity, short peptide, enzyme, and structure-supported classes. Also adds dynamic protein batching and masked pooling so low pLDDT can be interpreted by expected protein type.*

- [ ] **Track: Long-Range CodonLM Objectives**
*Link: [./tracks/long_range_codon_objectives_20260616/](./tracks/long_range_codon_objectives_20260616/)*
*Summary: Add config-gated multi-offset future-token losses, denoising/recovery training, and eventually structural auxiliary/preference losses so CodonLM learns longer-range protein constraints and becomes more robust to off-distribution generation.*
