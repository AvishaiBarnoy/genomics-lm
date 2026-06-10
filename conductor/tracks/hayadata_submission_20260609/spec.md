# hayaData 2026 Submission Track Specification

## 1. Overview
This track governs the preparation, drafting, review, and rehearsal of the `genomics-lm` talk proposals for the **hayaData 2026** conference. The primary objective is to select the best track/abstract, format it to Sessionize CFP standards, draft presentation slides, and prepare a local dry-run before the event.

---

## 2. Target Submission Options

### A. Track: MLOps / Practical ML
*   **Talk Title:** GPT on a Shoestring Budget: Solving Context-Length Limits with Generator-Critic Re-feeding on a Laptop
*   **Technical Angle:** Bypassing context-length RAM limitations (quadratic attention complexity causing OOM on 8GB laptops) by decoupling DNA-level causal generation (CodonLM) from protein-level function evaluation (ProteinLM).
*   **Key Concepts:** Generator-Critic loop, next-step biological re-feeding, Reset-and-Discard (ReD) sampling, Scaled Dot Product Attention (SDPA), gradient accumulation, version-based state caching.

### B. Track: Data Science & AI/ML
*   **Talk Title:** Probing the Hidden Mind of a Genomic GPT: How a 1D Language Model Learns 3D DNA Physics
*   **Technical Angle:** Representation probing of frozen transformer hidden states to prove that autoregressive models trained on 1D sequences implicitly learn 3D physical DNA shape properties (e.g., minor groove width, roll, electrostatic potential) and taxonomic dialects (codon usage bias).
*   **Key Concepts:** Linear probing, attention mapping, taxonomy bias analysis, DNAshapeR correlation.

---

## 3. Deliverables & Acceptance Criteria

1.  **CFP Submission Draft:** Finalized abstracts, takeaways, and speaker bio ready to submit to Sessionize.
2.  **Marp Slide Deck:** A markdown-based slide deck (compatible with Marp) outlining the story of "frugal biological AI."
3.  **Local Playground Demo:** The Streamlit web dashboard configured as a demo environment to visually showcase:
    *   Autoregressive generation (Start/Stop highlighting).
    *   Multi-task Protein Critic classification (stability, Pfam, EC).
    *   Attention weight maps.
4.  **Dry-Run Verification:** A recorded or timed rehearsal of the talk (targeting 30–40 minutes depending on the hayaData slot structure).
