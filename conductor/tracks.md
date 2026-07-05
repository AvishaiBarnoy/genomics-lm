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
*Summary: Physical-termination transfer pilot completed 3 epochs on a 24-GBFF hybrid CDS+UTR dataset. Validation improved (`val_loss` 5.496 → 4.882), but matched prefix generation still had 0% natural stops and 100% hard caps; median GQS degraded while local AA-prefix identity improved. Next signal should be generated-prefix replay/hard negatives, not more of the same teacher-forced objective.*

- [ ] **Track: Protein Latent Energy-Based Model (Stage 2.6)**
*Link: [./tracks/protein_ebm_20260610/](./tracks/protein_ebm_20260610/)*
*Summary: Implement a latent-space EBM for protein stability guided Langevin sampling, alongside token sliding-window Shannon entropy loop detection and EBM-guided early abort in ReD to optimize token yields.*

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

- [ ] **Track: CodonLM Trainer Refactor**
*Link: [./tracks/codonlm_trainer_refactor_20260622/](./tracks/codonlm_trainer_refactor_20260622/)*
*Summary: Opened to split `src/codonlm/train_codon_lm.py` into testable checkpoint/resume, data setup, objective computation, runtime loop, and CLI layers while preserving current commands, configs, checkpoint compatibility, and mid-epoch resume behavior.*

- [ ] **Track: Suite Runner Main.sh Evolution**
*Link: [./tracks/suite_runner_main_20260623/](./tracks/suite_runner_main_20260623/)*
*Summary: Opened to evolve `main.sh` from a CodonLM-specific wrapper into an explicit suite runner for CodonLM and ProteinLM workflows, with trainer-type dispatch, CodonLM backward compatibility, and no accidental ProteinLM use of CodonLM data prep/evaluation.*

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

- [x] **Track: PDB-Filtered Structural Fine-Tuning**
*Link: [./tracks/pdb_structural_finetuning_20260616/](./tracks/pdb_structural_finetuning_20260616/)*
*Summary: Completed. Curated bacteria UniProt-PDB subset mapped to GenBank CDS indices. Fine-tuned the best 69-token MLP Replay checkpoint on this structured subset for 5 epochs. Validation perplexity dropped to 56.83. Checked against sanity KPIs, showing a 100% natural stop rate and a 2x increase in genomic codon alignment, with physical DNAshape awareness fully conserved.*

- [ ] **Track: Structural-Aware ProteinCritic**
*Link: [./tracks/structural_aware_protein_critic_20260616/](./tracks/structural_aware_protein_critic_20260616/)*
*Summary: Protein-type labels, dynamic protein batching, masked pooling, safe transfer training, imbalance-aware `pos_weight`, and calibrated threshold/top-fraction evaluation are implemented. The weighted critic improves rare-label ranking but hurts raw probability calibration; keep open for generated-library rescoring and integration into selection loops.*

- [x] **Track: Long-Range CodonLM Objectives**
*Link: [./tracks/long_range_codon_objectives_20260616/](./tracks/long_range_codon_objectives_20260616/)*
*Summary: Designed and ablated Separate-Heads Multi-Offset MLP prior architectures and Generated-Prefix Replay training. Multi-offset 2-layer MLP heads combined with backbone freezing resolved next-token prior conflicts (ppl 59.51). Prefix replay training corrected generated-context drift, yielding a 100% natural stop rate (0% stalls) and a +115% alignment similarity (GQS) increase to 56.4. Detailed report: [docs/separate_heads_multi_offset_report.md](../docs/separate_heads_multi_offset_report.md) and [docs/replay_training_theory_and_results.md](../docs/replay_training_theory_and_results.md).*

- [ ] **Track: Bidirectional Backbone & Attention-Pooling for MultiTask ProteinCritic**
*Link: [./tracks/critic_bidirectional_attention_pooling_20260622/](./tracks/critic_bidirectional_attention_pooling_20260622/)*
*Summary: Integrate bidirectional attention into the ProteinCritic backbone, and replace average pooling with learnable attention-based pooling for active-site focus and saliency visualization.*

- [x] **Track: Architectural Upgrades (RoPE & SwiGLU) for CodonLM**
*Link: [./tracks/codonlm_architectural_upgrades_20260622/](./tracks/codonlm_architectural_upgrades_20260622/)*
*Summary: Completed. Implemented Rotary Position Embeddings (RoPE) and SwiGLU Gated Feed-Forward blocks inside the TinyGPT backbone. Verified shape/loss compatibility and legacy fallback mechanisms with unit tests. Evaluated performance via a 2x2 ablation study on MPS GPU, isolating a relative validation perplexity reduction of 2.5% for RoPE-only configurations.*

- [ ] **Track: Biophysical Regression & MLP Probes**
*Link: [./tracks/biophysical_mlp_regression_probes_20260622/](./tracks/biophysical_mlp_regression_probes_20260622/)*
*Summary: Implement non-linear MLP probes for categorical amino acid properties, and develop linear regression probes targeting continuous biophysical scales (hydropathy, isoelectric point, volume) to evaluate static codon embeddings.*

- [x] **Track: Non-Linear Offset Priors & Backbone Freezing (Stage 2.6)**
*Link: [./tracks/non_linear_offset_priors_20260702/](./tracks/non_linear_offset_priors_20260702/)*
*Summary: Integrate 2-layer MLP projection heads with GeLU for multi-offset targets, implement backbone/next-token head freezing during training to preserve baseline perplexity, and train for 5-10 epochs.*
>>>>>>> c812517 (feat(codonlm): implement non-linear MLP projection priors with parameter-efficient backbone)
