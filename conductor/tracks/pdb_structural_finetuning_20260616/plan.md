# PDB-Filtered Structural Fine-Tuning Plan

## Status

Open. Starter implementation added; no training run has been launched yet.

## Tasks

- [x] Define the structural training objective and acceptance criteria.
- [x] Add a CDS filtering utility that can prepare a structure-enriched subset from curated source line indices.
- [x] Add regression tests for the filtering utility.
- [x] Add a conservative Stage 3 fine-tuning config based on the best current `10L8H_d384` checkpoint.
- [ ] Enrich CDS extraction metadata with protein IDs, locus tags, and product/gene names so UniProt/PDB matching can be automated.
- [ ] Build a curated structure-positive line-index list from PDB/UniProt evidence.
- [ ] Tokenize and pack the filtered subset with `block_size=512`.
- [ ] Run the fine-tune and compare against Stage 2.6 using the structured-prefix and ESMFold evaluation harnesses.

## Initial Commands

```bash
python -m scripts.filter_cds_by_pdb \
  --dna data/processed/stage2.6_large_master_dna.txt \
  --meta data/processed/stage2.6_large_master_meta.tsv \
  --structured_line_indices data/processed/structured_pdb/line_indices.txt \
  --out_dir data/processed/structured_pdb
```

After filtering, tokenize and pack the subset with the existing codon tokenization and dataset builders, then train with:

```bash
python -m src.codonlm.train_codon_lm --config configs/stage3_structured_pdb_finetune.yaml
```

## Notes

This is the track that directly works toward a structural training signal. Critic-guided ReD and prefix prompting operate at inference time; this track changes the supervised next-codon distribution toward proteins with known structural support.
