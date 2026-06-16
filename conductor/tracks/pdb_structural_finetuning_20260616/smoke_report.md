# PDB-Filtered Structural Fine-Tuning Smoke Report

## Summary

Implemented the first usable structural training-signal path for CodonLM:

- CDS extraction now preserves protein/gene metadata for future corpora.
- Current Stage 2.6 CDS can be filtered immediately by exact translated protein sequence against UniProt structured-protein rows.
- A one-epoch Stage 3 smoke fine-tune completed from the Stage 2.6 `10L8H_d384` checkpoint.

## Dataset

Command:

```bash
python -m scripts.filter_cds_by_pdb \
  --dna data/processed/stage2.6_large_master_dna.txt \
  --meta data/processed/stage2.6_large_master_meta.tsv \
  --uniprot_tsv data/raw/uniprot_bacteria_50_512.tsv \
  --out_dir data/processed/structured_pdb
```

Result:

- Source CDS: 44,953
- Structure-positive exact translated-protein matches: 884
- Packed windows: 728 train, 100 val, 56 test

## Fine-Tune

The Stage 3 config was corrected to match the transfer checkpoint architecture. The current Stage 2.6 checkpoint uses full multi-head attention key/value tensors, so the fine-tune config must not set `n_kv_head: 4`.

Smoke run:

- Run id: `stage3_structured_pdb_smoke2`
- Epochs: 1
- Validation loss: 4.0701
- Validation perplexity: 58.564

## Generation And Structure Check

Structured-prefix smoke:

- 6 continuations from DHFR/FolA, TEM-1 beta-lactamase, and TPI/TIM-barrel-like prefixes
- 0/6 naturally terminated
- Mean ProteinCritic stability: 0.572
- Max ProteinCritic stability: 0.575

ESMFold top-3:

| rank | prefix | stability | pLDDT |
|---:|---|---:|---:|
| 1 | dhfr_folA | 0.575 | 0.354 |
| 2 | tpiA | 0.575 | 0.418 |
| 3 | dhfr_folA | 0.572 | 0.435 |

## Interpretation

The smoke validates the full technical path: structure-positive subset creation, packing, fine-tuning, generation, and ESMFold submission. The one-epoch smoke does not yet improve pLDDT over the previous structured-prefix baseline, so the next evidence step is a full 3-epoch fine-tune plus a matched ESMFold comparison at the same sample size as the earlier structured-prefix experiment.
