# Corrected Downstream Evaluation Readiness

This document records whether downstream datasets are eligible for evaluation
with the corrected genome-held-out CodonLM. A stronger checkpoint does not repair
an invalid downstream split.

## EC

**Status: blocked pending a separately sourced controlled corpus.**

The legacy EC preparation performs a random record-level split. The corrected
preflight instead joins EC annotations to the frozen pretraining
`cds_meta.tsv`/`cds_dna.txt`, preserves its genome assignments, clusters translated
proteins with MMseqs2, and requires test clusters to be absent from LM training.

On `corrected-codonlm-v1`, 6,617 EC-labelled records matched the frozen CDS
metadata, but all 6,617 belong to pretraining-train genomes. None belong to the two
pretraining-test genomes. Consequently, no held-out EC test class exists and no
corrected EC score should be reported from the legacy corpus.

Reproduce the gate:

```bash
python -m scripts.prepare_controlled_ec_dataset \
  --cds-meta data/processed/corrected/corrected-codonlm-v1/genome/cds_meta.tsv \
  --cds-dna data/processed/corrected/corrected-codonlm-v1/genome/cds_dna.txt \
  --uniprot-metadata data/processed/uniprot_metadata_full.csv \
  --out-dir data/processed/downstream/corrected-codonlm-v1/ec_genome_protein_holdout \
  --mmseqs-executable mmseqs \
  --min-protein-identity 0.3 \
  --min-coverage 0.8
```

The command fails closed and writes `split_report.json` with the class and split
counts. The next valid EC route is a separately frozen annotated corpus whose
accessions are audited against the LM pretraining manifest.

## AMR

**Status: controlled split and pretraining-overlap audit passed.**

The CARD protein-homolog dataset was prepared with an MMseqs2
protein-cluster-held-out split at 30% minimum identity and 80% coverage:

- 5,098 valid labelled records after filtering;
- eight internal-stop/ambiguous protein translations quarantined;
- 25 exact matches to LM-training CDSs quarantined before splitting;
- six drug classes, all present in train and test;
- 3,733 train and 1,285 test records;
- 185 protein clusters;
- achieved test fraction 25.61%;
- MMseqs2 version `18-8cc5c`.

Reproduce:

```bash
python -m scripts.prepare_amr_dataset \
  --out_dir data/processed/downstream/corrected-codonlm-v1/amr \
  --protocol protein_cluster_held_out \
  --mmseqs-executable mmseqs \
  --min-protein-identity 0.3 \
  --min-coverage 0.8 \
  --threads 4 \
  --seed 42 \
  --pretraining-cds-meta \
    data/processed/corrected/corrected-codonlm-v1/genome/cds_meta.tsv \
  --pretraining-cds-dna \
    data/processed/corrected/corrected-codonlm-v1/genome/cds_dna.txt
```

The generated `split_report.json` records input checksums, MMseqs2 version and
command, thresholds, class distributions, achieved split fraction, and cluster
counts.

The independent post-build audit compares all 1,285 AMR test records with 74,600
frozen LM-training CDSs. It passes with zero exact full-CDS duplicates. It reports
34 shared protein clusters, 98.60% nearest-protein hit coverage, median nearest
protein identity 37.2%, and 95th percentile identity 73.9%. These homology
statistics must accompany the AMR result because pretraining on related proteins
may help representation quality even though exact sequence memorization is
excluded.

```bash
python -m scripts.audit_downstream_pretraining \
  --cds-meta data/processed/corrected/corrected-codonlm-v1/genome/cds_meta.tsv \
  --cds-dna data/processed/corrected/corrected-codonlm-v1/genome/cds_dna.txt \
  --downstream-seqs \
    data/processed/downstream/corrected-codonlm-v1/amr/protein_cluster_held_out/test_amr_seqs.csv \
  --output \
    data/processed/downstream/corrected-codonlm-v1/amr/protein_cluster_held_out/pretraining_overlap_audit.json \
  --mmseqs-executable mmseqs \
  --minimap2-executable minimap2 \
  --threads 4
```
