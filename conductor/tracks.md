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

- [x] **Track: Multi-Scale Biophysical Architecture (Stage 2.6)**
*Link: [./tracks/multiscale_biophysics_20260609/](./tracks/multiscale_biophysics_20260609/)*
*Summary: Implemented stride-3 1D CNN Nucleotide Encoder downsampler and late-fusion zero-initialized shape projection in TinyGPT. Fine-tuned the joint model on 91k hybrid sequence windows. While stop triggers did not yet activate autoregressively, prefix ablation validation showed that injecting physical shape embeddings significantly stabilizes gene generation, increasing median GQS by ~25% (21.46 → 26.79 at k=3) without perplexity regressions.*

- [x] **Track: Protein Latent Energy-Based Model (Stage 2.6)**
*Link: [./tracks/protein_ebm_20260610/](./tracks/protein_ebm_20260610/)*
*Summary: EBM model trained to 0.42 val loss. Guidance ablation sweep verified that EBM-guided early abort saves 85.6% tokens. Implemented Manifold-Regularized Langevin Sampler (yielding +6.5% pLDDT improvement) and integrated it fully into the Streamlit Web Dashboard playground.*

- [x] **Track: Hybrid DNA-Protein Critic Benchmark (Stage 2.6)**
*Link: [./tracks/hybrid_critic_20260610/](./tracks/hybrid_critic_20260610/)*
*Summary: Implemented closed-loop guided generation (stability classifier and EBM energy) with Top-K candidate pruning. Benchmarked results showing EBM guidance reduces energy by 21.0 units while parallel GPU batching accelerates generation speed by 2.2x over baseline. Integrated controls into the Streamlit web dashboard.*

- [ ] **Track: Multi-Frame Overlapping Gene Modeling (Stage 3)**
*Link: [./tracks/multi_frame_overlapping_20260610/](./tracks/multi_frame_overlapping_20260610/)*

- [x] **Track: Progressive High-Capacity Scaling Ladder (Stage 2.7)**
*Link: [./tracks/progressive_scaling_20260610/](./tracks/progressive_scaling_20260610/)*
*Summary: Closed (Halted). Developed shape expansion tool and successfully validated shape upscaling. Pilot runs on unfrozen d512 models triggered silent macOS kernel OOM SIGKILLs, demonstrating that d512 backpropagation exceeds local M2 unified memory capacities. Established d384 as the local capacity ceiling.*

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
*Summary: All 5 phases implemented (GQA, mmap, BucketBatchSampler, CUDA device priority, SDPA path). The July 2026 MPS benchmark superseded the earlier batch-4 result: batch 8 with accumulation 16, AMP, checkpointing off, standard MHA, and 8 buckets reached about 5.5k useful tokens/second (about 2.0x the original baseline). Scientific quality parity remains to be established in the follow-up validation track.*

- [ ] **Track: Optimized Training Quality Validation & Context Ablation (Stage 2.6)**
*Link: [./tracks/optimized_training_validation_20260717/](./tracks/optimized_training_validation_20260717/)*
*Summary: Validate the measured MPS throughput winner with paired equal-token training runs, then compare 128-, 256-, and 512-token contexts using lossless chunking and token-budgeted updates before promoting new production defaults.*

- [x] **Track: CodonLM Trainer Refactor**
*Link: [./tracks/codonlm_trainer_refactor_20260622/](./tracks/codonlm_trainer_refactor_20260622/)*
*Summary: Completed. Refactored the monolithic 1,200-line `train_codon_lm.py` script into a modular `src/codonlm/training/` subpackage (separating config, checkpoints, objectives, and training loop) while preserving CLI backwards compatibility and mid-epoch resume semantics. Updated all import locations across scripts and tests.*

- [x] **Track: Suite Runner Main.sh Evolution**
*Link: [./tracks/suite_runner_main_20260623/](./tracks/suite_runner_main_20260623/)*
*Summary: Completed. Evolved `main.sh` into a multi-trainer suite runner supporting `codon_lm`, `protein_lm`, `protein_multitask`, and `protein_classifier` dispatches. Implemented config-based trainer resolution, fail-fast dataset existence checks, a `--preprocess-only` flag, and a `--dry-run` parameter.*

- [x] **Track: AMR Classification Probe (Conference)**
*Link: [./tracks/amr_classification_20260615/](./tracks/amr_classification_20260615/)*
*Summary: Completed. Created preparation script for CARD dataset. Integrated strict, homology-aware gene family splits (PR #66) and upgraded evaluations with Balanced Accuracy, Macro-AUPRC, and 1000-resample bootstrapped 95% confidence intervals (PR #69) to handle class imbalance.*

- [x] **Track: EC & AMR Downstream Evaluation (Conference)**
*Summary: Completed EC Level-1 probe (AUROC=0.703), homology-aware AMR probe (AUROC=0.893), k-mer baselines, UMAP+attention figures, SOTA table consolidation.*

- [x] **Track: Generative Design Loop**
*Link: [./tracks/generative_design_loop_20260615/](./tracks/generative_design_loop_20260615/)*
*Summary: ReD sampling + MultiTask ProteinCritic scoring + ESMFold API. Closes the generation→structure evaluation loop. 50/50 sequences terminated, pairwise identity 9.2%, ESMFold pLDDT ≈ 0.4–0.6 (novel/disordered — improvement direction: critic-guided ReD).*

- [x] **Track: Structured Protein Generation**
*Link: [./tracks/structured_generation_20260616/](./tracks/structured_generation_20260616/)*
*Summary: Closed as an experimental finding. Critic-guided ReD, family filtering, annealing, top-p sampling, and a structured-prefix harness were implemented; critic stability improved (+13.6%), but ESMFold pLDDT did not. Report: [./tracks/structured_generation_20260616/report.md](./tracks/structured_generation_20260616/report.md).*

- [x] **Track: PDB-Filtered Structural Fine-Tuning**
*Link: [./tracks/pdb_structural_finetuning_20260616/](./tracks/pdb_structural_finetuning_20260616/)*
*Summary: Completed. Curated bacteria UniProt-PDB subset mapped to GenBank CDS indices. Fine-tuned the best 69-token MLP Replay checkpoint on this structured subset for 5 epochs. Validation perplexity dropped to 56.83. Checked against sanity KPIs, showing a 100% natural stop rate and a 2x increase in genomic codon alignment, with physical DNAshape awareness fully conserved.*

- [x] **Track: Structural-Aware ProteinCritic**
*Link: [./tracks/structural_aware_protein_critic_20260616/](./tracks/structural_aware_protein_critic_20260616/)*
*Summary: Completed. Upgraded generative design loop with multi-label protein-type classification reporting and ESMFold calibration. Implemented step-wise Active ReD sequence complexity assertions to abort non-viable runs early (saving >80% compute). Evaluated baseline vs. fine-tuned generators under stability-filtering, confirming a +0.292 mean stability shift and a 3x yield increase in stable candidates.*

- [x] **Track: Long-Range CodonLM Objectives**
*Link: [./tracks/long_range_codon_objectives_20260616/](./tracks/long_range_codon_objectives_20260616/)*
*Summary: Designed and ablated Separate-Heads Multi-Offset MLP prior architectures and Generated-Prefix Replay training. Multi-offset 2-layer MLP heads combined with backbone freezing resolved next-token prior conflicts (ppl 59.51). Prefix replay training corrected generated-context drift, yielding a 100% natural stop rate (0% stalls) and a +115% alignment similarity (GQS) increase to 56.4. Detailed report: [docs/separate_heads_multi_offset_report.md](../docs/separate_heads_multi_offset_report.md) and [docs/replay_training_theory_and_results.md](../docs/replay_training_theory_and_results.md).*

- [x] **Track: Bidirectional Backbone & Attention-Pooling for MultiTask ProteinCritic**
*Link: [./tracks/critic_bidirectional_attention_pooling_20260622/](./tracks/critic_bidirectional_attention_pooling_20260622/)*
*Summary: Completed. Implemented a non-causal bidirectional attention encoder backbone with learnable Attention-Pooling and a shared latent bottleneck layer. Added an active-site motif saliency regularization loss to force focus on catalytic signatures. Convergence training on MPS GPU achieved stability classification accuracy of 79.23% (exceeding 77% target) and a 17.97x attention contrast ratio (exceeding 2.0x target).*

- [x] **Track: Architectural Upgrades (RoPE & SwiGLU) for CodonLM**
*Link: [./tracks/codonlm_architectural_upgrades_20260622/](./tracks/codonlm_architectural_upgrades_20260622/)*
*Summary: Completed. Implemented Rotary Position Embeddings (RoPE) and SwiGLU Gated Feed-Forward blocks inside the TinyGPT backbone. Verified shape/loss compatibility and legacy fallback mechanisms with unit tests. Evaluated performance via a 2x2 ablation study on MPS GPU, isolating a relative validation perplexity reduction of 2.5% for RoPE-only configurations.*

- [x] **Track: Biophysical Regression & MLP Probes**
*Link: [./tracks/biophysical_mlp_regression_probes_20260622/](./tracks/biophysical_mlp_regression_probes_20260622/)*
*Summary: Completed. Implemented continuous Ridge regression probes (targeting Kyte-Doolittle hydropathy, molecular weight, isoelectric point) and MLP classification probes on static embeddings. Comparative evaluation on baseline vs. advanced separate-heads prior models demonstrated a +0.085 increase in non-linear hydropathy accuracy and a 3x absolute Pearson correlation increase on residue charges (pI).*

- [x] **Track: Non-Linear Offset Priors & Backbone Freezing (Stage 2.6)**
*Link: [./tracks/non_linear_offset_priors_20260702/](./tracks/non_linear_offset_priors_20260702/)*
*Summary: Integrate 2-layer MLP projection heads with GeLU for multi-offset targets, implement backbone/next-token head freezing during training to preserve baseline perplexity, and train for 5-10 epochs.*

- [ ] **Track: Synonymous Generation Mode (Stage 2.6)**
*Link: [./tracks/synonymous_generation_20260707/](./tracks/synonymous_generation_20260707/)*
*Summary: Implement constrained decoding that maps amino acid sequences to synonymous DNA candidates.*

- [ ] **Track: Multi-Constraint Scoring & Optimization (Stage 2.6)**
*Link: [./tracks/multi_constraint_optimization_20260707/](./tracks/multi_constraint_optimization_20260707/)*
*Summary: Build scoring tracks for CAI, GC% deviation, local mRNA secondary structure, and sequence motif checks ( forbidden sites / repeats).*

- [ ] **Track: Scientific Inquiry: Prefix-to-Function Experiment (Stage 2.6)**
*Link: [./tracks/prefix_to_function_inquiry_20260707/](./tracks/prefix_to_function_inquiry_20260707/)*
*Summary: Investigate context saturation by measuring functional classification agreement as a function of prefix length.*
