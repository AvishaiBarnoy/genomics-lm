# Investigation: Secondary mRNA Structural Motifs as Termination Cues in CodonLM

This report evaluates whether the **CodonLM** model has learned to utilize secondary structure motifs (such as GC-rich hairpins and Poly-T/U tracts) as physical landmarks for gene termination before it has fully mastered the frame-dependent, semantic placement of stop codons.

---

## 1. Biological Foundation & Hypothesis

In prokaryotic genomes, transcription termination occurs via two main pathways:
1. **Rho-dependent termination**: Requires the Rho protein factor.
2. **Rho-independent (intrinsic) termination**: Driven entirely by the physical properties of the transcribed mRNA sequence:
   * **GC-rich Hairpin (Stem-Loop)**: A stable dyad symmetry (inverted repeat) in the mRNA folds into a stem-loop. This hairpin physically interacts with the RNA polymerase (RNAP), causing it to pause.
   * **Poly-U Tract**: A run of 6-8 Uracils (encoded as `TTTTTT` in coding DNA) immediately downstream of the hairpin. The hydrogen bonding between the rU of the transcript and dA of the DNA template is extremely weak, enabling the paused transcript to easily dissociate.

### The Hypothesis
Since we know from previous stages that CodonLM's hidden states strongly correlate with local 3D DNA physical properties (such as Electrostatic Potential, Roll, and Minor Groove Width), the model might use these frame-independent physical cues (local GC-richness, dyad symmetry, and poly-T tracts) as "landmarks" to trigger a transition to a boundary/stopping state (generating a Stop codon or `<EOS_CDS>`), even before it learns the complex, frame-tracking semantics of stop-codon placement.

---

## 2. Analytical Framework & Implementation

To test this hypothesis, we built and executed two distinct analytical probes on the **Stage 2.5 model** (`2026-06-06_stage2.5_6L4H_d256_e20`):

### Probe A: Empirical Sampling Analysis (`check_termination_motifs.py`)
We ran open-ended generation from 150 real coding sequence (CDS) prefixes at $T=1.0$ and $\text{top-k}=0$ to allow natural, early-terminating trajectories. We extracted the 30 bp window immediately preceding the termination point for:
* **Early Terminated Sequences**: Sequences that generated a stop codon (`TAA`/`TAG`/`TGA`) or `<EOS_CDS>` before reaching the 60-codon target.
* **Hard Capped Sequences**: Sequences that reached the 60-codon target without terminating.

For both groups, we measured:
1. **Hairpin Score**: A thermodynamic heuristic counting complementary base pairings (A-T = 2, G-C = 3, mismatch = -1) across loop sizes of 3-9 nt.
2. **Poly-T Run length**: The longest run of consecutive Ts/Us.
3. **DNAshape parameters**: Average minor groove width (MGW), roll, and electrostatic potential (EP).

### Probe B: In-Silico Perturbation / Causal Probing (`test_perturbation_motifs.py`)
To bypass sampling noise and directly test the causal sensitivity of the model, we took 100 real CDS prefixes and appended 4 synthetic tails (all 30 bp, codon-aligned):
1. **Control (Neutral)**: A neutral, GC-balanced alanine-rich sequence (`GCGGCGGCCGCCGCAGCGGCGGCCGCCGCA`).
2. **Poly-T Tract**: A tail ending in a long run of Ts (`GCGGCGGCCGCCGCATTTTTTTTTTTTTTT`).
3. **Hairpin Stem-Loop**: A tail forming a highly stable GC-rich stem-loop (`GCGGCCGCGGAAAAAACCGCGGCCGCGGCG`).
4. **Full Intrinsic Terminator**: A tail combining the stable GC-rich hairpin directly followed by a poly-T tract (`GCCGCGGCCGCGAAAACCGCGGCCGCTTTT`).

We then evaluated the model's next-token logits to compute the exact probability of transition to a Stop codon or the `<EOS_CDS>` token immediately following the tail.

---

## 3. Empirical Results

### Probe A: Sampling Distribution (150 Samples)
Under natural sampling, we observed an early-termination rate of **74.7%** (112 early terminated vs. 38 hard capped). The metrics of the 30 bp window preceding the termination/end point are summarized below:

| Metric | Early Terminated (N=112) | Hard Capped (N=38) | Difference |
| :--- | :---: | :---: | :---: |
| **Avg Hairpin Score** | 12.08 (max=19.0) | 11.97 (max=17.0) | **+0.11** |
| **Avg Poly-T Run** | 1.92 (max=4) | 1.92 (max=6) | **0.00** |
| **Poly-T Run $\ge$ 4 Frac** | 3.6% | 2.6% | **+1.0%** |
| **Average MGW** | 4.52 Å | 4.53 Å | **-0.01 Å** |
| **Average Roll** | 3.18° | 3.41° | **-0.23°** |
| **Average EP** | -4.99 kT/e | -5.00 kT/e | **+0.01 kT/e** |

> [!NOTE]
> The physical properties and structural motif scores are nearly identical between early-terminated and hard-capped sequences, indicating no statistical enrichment of termination motifs immediately preceding early stops.

---

### Probe B: Causal Perturbation (100 Prefixes)
Evaluating the exact next-token probabilities at the end of the synthetic tails yielded the following transition probabilities:

| Variant | Stop Codon Prob | `<EOS_CDS>` Prob | Total Termination Prob |
| :--- | :---: | :---: | :---: |
| **Control (Neutral)** | 2.0873% | 0.0000% | **2.0873%** |
| **Poly-T Tract** | 2.1555% | 0.0000% | **2.1555%** |
| **Hairpin Stem-Loop** | 2.1254% | 0.0000% | **2.1254%** |
| **Full Terminator** | 2.1448% | 0.0000% | **2.1448%** |

> [!IMPORTANT]
> 1. **No Causal Shift**: Appending a stable hairpin or a full terminator only shifted the stop codon probability by a negligible margin (~0.04% to 0.07%).
> 2. **Strict Grammar for EOS**: The probability of emitting `<EOS_CDS>` directly was **exactly 0.0000%** across all tails. The model has learned a strict grammar where `<EOS_CDS>` must be immediately preceded by a stop codon, and it does not jump to it based on physical structure alone.

### Probe C: Post-STOP UTR Generation Analysis (`test_utr_generation.py`)
To test if the model generates these motifs *after* it transitions past the Stop codon, we extracted 100 biological stop-codon boundaries (genes + 30 bp of downstream UTR) from our validation set. We prompted the model with the sequence up to and including the stop codon, generated the next 30 bp (10 codons), and compared them to ground-truth UTRs and mid-CDS control generations:

| Metric | Biological Ground Truth UTR | Generated Post-STOP UTR | Mid-CDS Control Generation |
| :--- | :---: | :---: | :---: |
| **Avg Hairpin Score** | 11.72 | 12.79 | 11.91 |
| **Avg Poly-T Run** | 2.47 (max=6) | 1.82 (max=5) | 1.96 (max=5) |
| **Poly-T Run $\ge$ 4 Frac** | **17.9%** | **1.3%** | **5.1%** |

#### Interpretation of Probe C:
* **No UTR Poly-T Generation**: While **17.9%** of native biological UTRs directly following stop codons contain a poly-T run of length $\ge 4$ (due to native Rho-independent terminators), only **1.3%** of the model's generated post-STOP sequences contain one (even lower than the random background mid-CDS control of 5.1%).
* **No Stable Hairpin Generation**: The generated post-STOP hairpin score (~12.79) remains close to the random noise baseline (~12.0) and mid-CDS controls, showing no thermodynamic enrichment for stable stem-loops.
* **Conclusion**: The model does not generate these motifs in UTR spaces. Therefore, we **cannot** use them for manual termination because the model is not producing them.

---

## 4. Key Findings & Biological Interpretation

Our empirical tests demonstrate that **CodonLM has NOT yet learned to generate or associate mRNA secondary structures (hairpins/poly-T tracts) with termination boundaries**. The model's early termination behavior remains purely stochastic or driven by local codon usage rather than these physical cues. 

### Why the Model Hasn't Learned This Yet:
1. **Translation vs. Transcription Termination**: In biology, stop codons (translation termination) are recognized by protein release factors, not by RNA hairpins. Intrinsic terminators (hairpin + poly-T) terminate transcription (RNA polymerase) and reside in the **3' UTR** (non-coding intergenic regions), which are downstream of the stop codon.
2. **Left-to-Right Causal Structure**: Because the model generates left-to-right, it must decide to output a Stop codon *before* it generates the downstream UTR containing the hairpin and poly-T tract. The cue must be present in the coding sequence itself (e.g., protein length limits or translation rates), whereas transcription terminators appear *after* the coding sequence ends.
3. **CDS-Centric Bias**: The training set focuses predominantly on coding regions (CDS). Even with Genomic Tapes and Anchored Operon Bridges, the model's vocabulary and context are highly biased toward in-frame coding triplets.
4. **Tokenization Limitation**: Codon tokenization groups nucleotides into chunks of 3. However, secondary structure folding is extremely sensitive to individual nucleotide base pairings. A single nucleotide shift can disrupt a stem-loop, a detail that is heavily smoothed over by codon-level tokenization.

---

## 5. Roadmap Recommendations

To enable the model to learn and utilize structural termination cues, we recommend the following modifications to the genomics-lm roadmap:

> [!TIP]
> ### 1. Move to a Nucleotide-Level LM (Stage 4)
> Moving from a codon vocabulary (64 codons + specials) to a nucleotide vocabulary (A, C, G, T + specials) will allow the model's self-attention layers to directly map base-pairing symmetries at the single-nucleotide level, which is crucial for capturing hairpins.
>
> ### 2. Hard-Mine 3' UTR Regions for Training
> Instead of anchoring only on the Stop-to-Start boundary, construct training windows centered on the **Stop-to-Transcript-End** boundaries (capturing the full 3' UTR containing the native Rho-independent terminators).
>
> ### 3. Incorporate Free Energy ($\Delta G$) Loss Regularization
> During generation, we could guide the generator using the Multi-Task Critic by adding an auxiliary reward/loss representing the Minimum Free Energy (MFE) of the 3' UTR sequence, forcing the model to generate physically viable terminator structures.
