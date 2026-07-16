# Multi-Constraint Scoring & Optimization Plan

## Status
Planned.

## Objectives
1. **Multi-Constraint Scoring Suite**: Implement evaluations for generated DNA sequences, including:
   - **Codon Adaptation Index (CAI)**: Host-specific codon bias matching.
   - **GC Content Constraints**: Scoring deviation from target GC ranges (local and global).
   - **Local mRNA Folding Energy**: Estimate stability ($\Delta G$) around the 5′ ribosomal entry region.
   - **Synthesis Constraints**: Detect forbidden restriction sites (e.g. EcoRI, HindIII, XbaI) and homopolymers (e.g., repeating A/T blocks).
2. **Pareto Candidate Ranking**: Implement a multi-objective scoring utility that returns a ranked list of candidate designs.
3. **Web Dashboard Dashboard Integration**: Display a comparative candidate table with scoring tracks in the Streamlit UI.
4. **Validation**: Write comprehensive unit tests verifying all scoring tracks output accurate, expected values.

## Tasks
- [ ] Implement a CAI calculator using target bacterial host codon usage matrices.
- [ ] Implement a GC content scoring module (both global GC and local sliding window GC).
- [ ] Implement a lightweight local mRNA secondary structure folding estimator (e.g. wrapper around ViennaRNA or an entropy-based proxy).
- [ ] Implement sequence constraint checking (homopolymer repeats and restriction site matching).
- [ ] Implement a multi-objective ranking algorithm (e.g. weighted score sum or Pareto-front utility).
- [ ] Expose candidates table in the web dashboard showing scoring metrics.
- [ ] Add unit tests verifying constraint check rules and calculation functions.
