# Multi-Frame Overlapping Gene Modeling Specification

## Overview
Implement multi-frame codon-level training to handle overprinted (overlapping) genes in high-density bacterial and viral genomes. Standard codon models are restricted to Frame 0, but prokaryotes often pack functional genes in alternate reading frames (+1, +2) on the same sequence span.

## Requirements
- **Multi-Frame Tokenizer:** Support parallel tokenization of sequences in all three forward reading frames (Frame 0, +1, +2) and raw nucleotides.
- **Alternating Context Embeddings:** Modify `TinyGPT` to accept reading frame identifier embeddings injected into positional representations.
- **Joint Probability Modeling:** The model must predict next-codon distributions for all three frames simultaneously, resolving frame-shifted overlaps.
