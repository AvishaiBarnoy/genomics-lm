# PDB-Filtered Structural Fine-Tuning Plan

## Status

Open. Metadata enrichment and sequence-based UniProt filtering are implemented.
A one-epoch smoke fine-tune completed successfully; a full fine-tune and larger
ESMFold comparison remain open.

## Tasks

- [x] Define the structural training objective and acceptance criteria.
- [x] Add a CDS filtering utility that can prepare a structure-enriched subset from curated source line indices.
- [x] Add regression tests for the filtering utility.
- [x] Add a conservative Stage 3 fine-tuning config based on the best current `10L8H_d384` checkpoint.
- [x] Enrich CDS extraction metadata with protein IDs, locus tags, gene names, product names, translations, db_xref, coordinates, and strand.
- [x] Add automatic structure-positive filtering by exact translated CDS sequence against UniProt `Sequence` rows with `3D-structure`/PDB evidence.
- [x] Tokenize and pack the filtered subset with `block_size=512`.
- [x] Run a one-epoch smoke fine-tune and compare with the structured-prefix + ESMFold harness.
- [x] Run the full 3-epoch fine-tune.
- [x] Run a larger matched comparison against Stage 2.6 (critic-scored prefix benchmarks).
- [ ] Run a matched ESMFold comparison (pending network-based structure prediction).

## Results (3-Epoch Full Fine-Tune)

- Run id: `2026-06-16_stage3_10L8H_d384_e3`
- Validation loss: 4.0681 (Baseline 2.6: 4.088)
- Validation perplexity: 58.45 (Baseline 2.6: 59.75)
- Mean ProteinCritic stability (prefixes): 0.572 (Baseline 2.6: 0.573)
- Mean Pfam Family confidence: 0.0478 (Baseline 2.6: 0.0458)
- Summary: The fine-tune successfully lowered validation loss/perplexity on the structure-positive subset. While sequence-based ProteinCritic stability remains stable, Pfam family confidence shows a slight upward trend. Structural validation (ESMFold) is the next definitive step.

| Metric (Prefixes) | Stage 2.6 Baseline | Stage 3 Fine-Tune |
|---|---:|---:|
| Median GQS (k=1) | 23.90 | 24.53 |
| Mean stability_prob | 0.573 | 0.572 |
| Mean Pfam confidence | 0.0458 | 0.0478 |

## Reproduction Commands


```bash
python -m scripts.filter_cds_by_pdb \
  --dna data/processed/stage2.6_large_master_dna.txt \
  --meta data/processed/stage2.6_large_master_meta.tsv \
  --uniprot_tsv data/raw/uniprot_bacteria_50_512.tsv \
  --out_dir data/processed/structured_pdb
```

After filtering, tokenize and pack the subset:

```bash
python -m src.codonlm.codon_tokenize \
  --inp data/processed/structured_pdb/cds_dna.txt \
  --out_ids data/processed/structured_pdb/codon_ids.txt \
  --out_vocab data/processed/structured_pdb/vocab_codon.txt \
  --out_itos data/processed/structured_pdb/itos_codon.txt \
  --termination eos

python -m src.codonlm.build_dataset \
  --ids data/processed/structured_pdb/codon_ids.txt \
  --group_meta data/processed/structured_pdb/cds_meta.tsv \
  --block_size 512 \
  --windows_per_seq 1 \
  --val_frac 0.1 \
  --test_frac 0.1 \
  --out_dir data/processed/structured_pdb_pack \
  --pack_mode single
```

Train with:

```bash
python -m src.codonlm.train_codon_lm --config configs/stage3_structured_pdb_finetune.yaml
```

## One-Epoch Smoke Results

- Filtered subset: 884 / 44,953 CDS exact translated-protein matches to structured UniProt rows.
- Packed windows: 728 train, 100 val, 56 test at `block_size=512`.
- One-epoch smoke run: `runs/stage3_structured_pdb_smoke2`.
- Smoke validation: `val_loss=4.0701`, `ppl=58.564`.
- Structured-prefix smoke: 6 generated continuations, 0/6 terminated, mean ProteinCritic stability `0.572`, max `0.575`.
- ESMFold top-3 pLDDT: `0.354`, `0.418`, `0.435`.

The smoke verifies that the structural subset and fine-tuning loop work, but it does not yet show a pLDDT improvement over the prior structured-prefix baseline. A larger/full fine-tune is still needed before drawing a strong conclusion.

## Notes

This is the track that directly works toward a structural training signal. Critic-guided ReD and prefix prompting operate at inference time; this track changes the supervised next-codon distribution toward proteins with known structural support.
