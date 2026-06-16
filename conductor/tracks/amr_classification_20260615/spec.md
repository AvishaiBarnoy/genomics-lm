# Track: AMR Classification Probe
**Track ID:** `amr_classification_20260615`
**Status:** 🟡 PLANNING
**Priority:** Medium (conference readiness)
**Depends on:** Stage 2.6 checkpoint (✅ done), EC probe pipeline (✅ done)

---

## Objective

Benchmark the CodonLM Stage 2.6 embeddings on **Antimicrobial Resistance (AMR) gene classification** — predicting whether a bacterial gene confers resistance to a specific antibiotic class.

This is the natural companion to the EC probe:
- **EC** tests *enzymatic function* separability
- **AMR** tests *resistance phenotype* separability — clinically significant and highly publishable

---

## Scientific Motivation

AMR is one of the most critical public health challenges globally. A language model that can linearly decode AMR phenotype from sequence embeddings would have direct clinical applications and is compelling for conference audiences.

Expected result range based on EC results:
- LogReg/SVM accuracy: **50–70%** (fewer classes than EC, more biologically coherent)
- If < 40%: embeddings do not capture AMR signal → negative result worth reporting
- If ≥ 60%: strong positive result, publishable finding

---

## Dataset Plan

### Primary Source: CARD (Comprehensive Antibiotic Resistance Database)
- **URL:** https://card.mcmaster.ca/download
- **File:** `broadstreet_v3.x.x.tar.bz2` → `protein_fasta_protein_homolog_model.fasta`
- **Labels:** AMR Gene Family → Antibiotic class (β-lactam, Tetracycline, Aminoglycoside, Fluoroquinolone, Macrolide, etc.)
- **Format:** FASTA with metadata in header line (parseable)
- **License:** CC BY 4.0 (free for academic use)

### Preprocessing Steps
1. Download CARD nucleotide sequences (coding sequences for resistance genes)
2. Parse `aro_index.tsv` to extract: `ARO_accession`, `AMR Gene Family`, `Drug Class`, `Resistance Mechanism`
3. Filter to **top-N antibiotic classes** (≥ 50 gene examples) — target 5–8 classes
4. Extract CDS sequences → tokenize with existing CodonLM tokenizer
5. Extract embeddings via `scripts/extract_embeddings.py` (mean-pool hidden states)
6. Save as `data/labels/train_amr.csv` and `data/labels/test_amr.csv`

### Alternative Source: NCBI AMRFinderPlus
- Curated AMR gene reference catalog
- Available at: https://ftp.ncbi.nlm.nih.gov/pathogen/Antimicrobial_resistance/AMRFinderPlus/database/latest/

---

## Implementation Plan

### Phase 1: Data Acquisition & Preprocessing (≈ 2h)
- [ ] Download CARD `broadstreet` nucleotide FASTA
- [ ] Write `scripts/prepare_amr_dataset.py`:
  - Parse CARD FASTA headers → drug class labels
  - Filter to sequences 50–512 codons (fits model context)
  - Stratified train/test split (80/20)
  - Output: `data/labels/train_amr.csv`, `data/labels/test_amr.csv`
- [ ] Validate class distribution (print histogram)

### Phase 2: Embedding Extraction (≈ 15 min)
- [ ] Run `scripts/extract_embeddings.py` with Stage 2.6 checkpoint on AMR sequences
- [ ] Output: `data/embeddings/amr_train_embeddings.npy`, `amr_test_embeddings.npy`

### Phase 3: Probe Training (≈ 5 min)
- [ ] Create `configs/classifier/probe_amr.yaml` (copy from `probe_ec.yaml`, update paths/labels)
- [ ] Run `scripts/train_classifier.py --config configs/classifier/probe_amr.yaml`
- [ ] Evaluate: LogReg, SVM, MLP — same protocol as EC

### Phase 4: Results & Reporting
- [ ] Add AMR results row to `conference/sota_benchmark_table.md`
- [ ] Update `docs/DEVELOPMENT_LOG.md`
- [ ] Add `amr` column to `runs/_summary/summary.csv`

---

## Expected Outputs

| File | Description |
|---|---|
| `data/labels/train_amr.csv` | AMR training labels (gene CDS + antibiotic class) |
| `data/labels/test_amr.csv` | AMR test labels |
| `configs/classifier/probe_amr.yaml` | Probe training config |
| `runs/2026-06-15_stage2.6_10L8H_d384_e10/scores/metrics.json` | Updated with `amr_*` keys |
| `conference/sota_benchmark_table.md` | Updated with AMR column |

---

## Antibiotic Class Target Labels

| Class | Mechanism | Expected # genes in CARD |
|---|---|---|
| β-lactam | β-lactamase production | ~800 |
| Tetracycline | Efflux pump / ribosome protection | ~300 |
| Aminoglycoside | Enzymatic modification | ~400 |
| Fluoroquinolone | Target alteration / efflux | ~250 |
| Macrolide | Ribosomal methylation / efflux | ~200 |
| Colistin | Membrane modification | ~100 |

---

## Risk Assessment

| Risk | Mitigation |
|---|---|
| CARD sequences in protein (AA) not nucleotide | Use `nucleotide_fasta_protein_homolog_model.fasta` specifically |
| Sequences too long (>512 codons) | Truncate or skip — document threshold |
| Class imbalance (β-lactam >> others) | Use Macro-F1 + AUROC as primary metrics; stratified split |
| AMR not linearly separable from CodonLM | Report negative result; meaningful scientific finding |
