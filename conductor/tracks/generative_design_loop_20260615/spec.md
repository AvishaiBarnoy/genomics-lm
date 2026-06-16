# Track: Generative Design Loop

**Date:** 2026-06-15
**Status:** In Progress

---

## Goal

Close the **generation → evaluation** loop:

1. **CodonLM** generates candidate DNA/codon sequences using ReD sampling (Reset-and-Discard)
2. **MultiTask ProteinCritic** scores each translated AA sequence on Pfam family, EC function, and stability
3. **ESMFold API** (optional) predicts 3D structure and returns pLDDT confidence for the top-K sequences

This track provides the missing **"generative"** result for the conference abstract, demonstrating that the model can produce novel, functionally-scored sequences — not just classify existing ones.

---

## Components

| Component | Description |
|---|---|
| **ReD Sampling** | Reset-and-Discard: restart generation until a properly stop-codon-terminated sequence is produced |
| **MultiTask ProteinCritic** | Simultaneous scoring on Pfam family (1 000 classes), EC function (500 classes), and stability (binary) |
| **Diversity Metrics** | Pairwise sequence identity, k-mer coverage, GC content per sequence |
| **ESMFold API** | Optional: submits top-K AA sequences to `api.esmatlas.com`; returns pLDDT mean/min/max |

---

## Key Script

```
scripts/generative_design_loop.py
```

**Module-level functions (all importable):**
- `translate_dna(dna)` — codon → AA translation with stop detection
- `red_generate(model, device, stoi, itos, ...)` — ReD sampling loop
- `pairwise_identity(seqs)` — average pairwise AA identity
- `kmer_diversity(seqs, k=3)` — fraction of 20^k possible k-mers observed
- `gc_content(codon_seqs)` — GC% per sequence

---

## Outputs

| File | Description |
|---|---|
| `outputs/reports/generative_design/design_library.csv` | Per-sequence record: codons, AA seq, critic scores, GC%, diversity, ESMFold pLDDT |
| `outputs/reports/generative_design/design_report.md` | Markdown summary with aggregate stats, top-10 sequences by stability, diversity analysis |

---

## Conference Value

- **Closes the generation → structure loop**: sequences are generated, functionally scored, and optionally structure-predicted end-to-end
- **Provides the missing "generative" result**: moves the abstract claim from "we can classify" to "we can generate and evaluate"
- **Diversity analysis**: demonstrates the model produces varied, non-trivial outputs rather than mode collapse
