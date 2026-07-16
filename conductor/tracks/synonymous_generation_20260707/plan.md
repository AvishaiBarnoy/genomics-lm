# Synonymous Generation Mode Plan

## Status
Planned.

## Objectives
1. **Synonymous Constrained Decoding**: Implement a decoding policy that accepts an amino acid sequence and forces the generator to select only from synonymous codons corresponding to the target residues at each step.
2. **Probability Normalization**: Re-normalize sampling probabilities (with temperature and top-p support) exclusively over the synonymous subset of the vocabulary.
3. **Streamlit UI Integration**: Expose the constrained synonymous generator in the visual Model Playground tab of the Streamlit dashboard, letting users input protein sequences.
4. **Validation**: Add unit tests verifying that all generated DNA sequences translate 100% identically to the input protein sequence.

## Tasks
- [ ] Implement synonymous vocabulary mapping from codon token indexes to standard amino acid abbreviations.
- [ ] Write the constrained sampling method in `src/codonlm/generate.py` or a dedicated module.
- [ ] Add unit tests verifying synonymous translation identity.
- [ ] Add temperature and top-p support to synonymous sampling.
- [ ] Integrate the constrained generator into the Streamlit dashboard (`scripts/web_dashboard.py`).
- [ ] Run benchmark generations for a set of target protein sequences and confirm performance under local memory constraints.
