# Conference Figures — Descriptions & Scientific Interpretation

Generated from: **CodonLM Stage 2.6** (`2026-06-15_stage2.6_10L8H_d384_e10`)
Architecture: 10-layer · 8-head · d=384 · 20.6M parameters
Training: 15 diverse bacterial genomes · 9 epochs · Apple M2 8GB MPS

Scripts: [`scripts/conference_umap.py`](../scripts/conference_umap.py) · [`scripts/conference_attention.py`](../scripts/conference_attention.py)

---

## Figure 1 — UMAP Codon Embedding Space

**File:** `figures/fig1_umap_codon_embeddings.png`

**What it shows:**
A 2D UMAP projection (cosine distance) of the 65 learned token embeddings — one per sense codon, plus 4 special tokens (`<PAD>`, `<BOS_CDS>`, `<EOS_CDS>`, `<SEP>`).

**How to read it:**
- Each **point = one codon** (3-nucleotide word from the vocabulary)
- **Color = amino acid identity** — synonymous codons encoding the same amino acid share the same color
- **ATG (★, cyan)** = start codon; **TAA/TAG/TGA (✕, red)** = stop codons
- Special tokens appear in grey, typically isolated from the sense codon cluster

**Key finding:**
Synonymous codons cluster together in embedding space — an **emergent structure learned purely from next-codon prediction loss**, without any structural or functional supervision. The start codon (ATG) and stop codons are geometrically separated from the sense codon cloud, reflecting their role as functional boundary tokens. This is direct evidence that the model has internalized the genetic code's degeneracy structure.

**Conference talking point:**
> "Without ever being told what an amino acid is, the model learned to group synonymous codons — those that code for the same protein building block — into coherent clusters. This shows the model has discovered the genetic code's degeneracy as a statistical regularity in the training data."

---

## Figure 2a — Attention Head Specialization Overview Grid

**File:** `figures/fig2a_attn_overview.png`

**What it shows:**
A 3-panel heatmap across all 10 layers × 8 heads showing three specialization metrics:
1. **Avg Entropy ↓** (left): Lower entropy = sharper, more focused attention distribution
2. **ATG Attention Bias ↑** (center): Average attention weight directed toward start codon positions
3. **Stop Codon Bias ↑** (right): Average attention directed toward stop codon positions

**Key finding:**
Layer 0 heads are consistently the most focused (lowest entropy) and carry the highest ATG bias. The cyan box marks the single best head in each panel. **L0·H4** dominates both the entropy and ATG-bias panels — it is simultaneously the most focused head *and* the best start-codon specialist, suggesting a dedicated gene-start recognition circuit in the earliest layer.

**Conference talking point:**
> "Early layers develop specialized boundary-detection heads. Head 4 in Layer 0 acts as a global ATG anchor — it appears to learn: 'I need to track where this gene began.' This is analogous to the [CLS] token mechanism in BERT-style models, but discovered here without any explicit architectural inductive bias."

---

## Figure 2b — Most Focused Attention Head (L0·H4, Rank #1)

**File:** `figures/fig2b_attn_head_focused_rank1.png`
*(Also: `fig2b_attn_head_focused_rank2.png` for rank #2 = L0·H1)*

**What it shows:**
Full (T×T) attention weight matrix for the most focused head across the first 60 tokens of a representative validation sequence. Cyan lines = ATG positions; red lines = stop codon positions.

**Key finding:**
The structured diagonal pattern (local context) with visible off-diagonal spikes specifically at ATG and stop codon columns indicates this head is **not** acting as a simple local n-gram detector. The cross-sequence spikes at functional boundaries confirm long-range boundary detection.

---

## Figure 2c — Start-Codon Specialist Head

**File:** `figures/fig2c_attn_start_specialist.png`

**What it shows:**
Attention heatmap for the head with the **highest average attention weight on ATG columns** across the entire sequence (L0·H4, bias = 0.0208).

**Key finding:**
A clear global retrieval pattern: any position in the sequence attends back to the ATG start codon position. This acts as a **"where did I begin?" positional anchor** — maintaining awareness of the gene start throughout the entire sequence. This mechanism is biologically meaningful: prokaryotic ribosomes similarly track the Shine-Dalgarno / start codon as a reference frame.

---

## Figure 2d — Attention Bias Across All 80 Heads

**File:** `figures/fig2d_attn_bias_barchart.png`

**What it shows:**
Bar chart of ATG (start codon, top) and stop codon (bottom) attention bias for all 80 heads (10L × 8H), sorted by start-codon bias. Cyan bars = top-8 most specialized heads.

**Key finding:**
A **sparse specialization pattern** — a small number of heads carry most of the functional boundary signal, with the majority near the mean. This is characteristic of learned interpretable circuits in transformers (cf. Olah et al., 2020; Elhage et al., 2021) and supports a mechanistic interpretability analysis as a future direction.

---

## Reproducibility

All figures are deterministically reproduced by:
```bash
conda activate codonlm
python -m scripts.conference_umap 2026-06-15_stage2.6_10L8H_d384_e10
python -m scripts.conference_attention 2026-06-15_stage2.6_10L8H_d384_e10
```
Output: `conference/figures/fig1_*.png`, `conference/figures/fig2*.png`
