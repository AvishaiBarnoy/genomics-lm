# Specification: Genomic Tape Extraction

## Overview
This track transitions the project from "Gene-based" extraction (individual CDS) to "Genomic Tape" extraction. By processing contiguous chromosomal segments, the model will see intergenic regions, promoters, and the polycistronic structure (operons) characteristic of prokaryotic biology.

## Objectives
1.  **Context Preservation:** Extract raw genomic sequences in contiguous blocks of 512-1024 codons regardless of gene boundaries.
2.  **Operon Logic:** Enable the model to learn the signals that coordinate multiple genes in a single transcript.
3.  **Regulatory Awareness:** Capture non-coding regulatory elements (promoters, operators, terminators) that are currently lost in single-CDS extraction.

## Technical Plan
- **Script:** Create `src/codonlm/extract_genomic_tape.py`.
- **Logic:** Slide a window across the entire chromosome, outputting fixed-length codon sequences.
- **Metadata:** Track the coordinates and strand information for each "tape" segment.
