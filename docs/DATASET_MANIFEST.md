# Versioned Dataset Manifest

Corrected CodonLM datasets use the `codonlm_dataset_manifest` schema. Version 1
binds source snapshots, split policy, ambiguity handling, lossless chunking,
packing, vocabulary, leakage gates, seeds, and derived artifacts into one
content-addressed dataset identity.

Validate a prepared dataset before training or evaluation:

```bash
python -m scripts.validate_dataset_manifest \
  data/processed/global/<DATASET_ID>/manifest.json
```

Validation fails on unsupported schema versions, inconsistent scientific-validity
claims, overlapping split groups, count mismatches, missing or modified sources,
missing or modified artifacts, untracked NPY sidecars, invalid special-token IDs,
or dataset tokens outside the declared vocabulary.

## Dataset Identity

`dataset.id` is the SHA-256 digest of deterministic JSON containing scientific
policies and content hashes. Storage paths and legacy compatibility fields are
excluded, so moving byte-identical sources and artifacts does not change the ID.
Changing a source, split, vocabulary, transformation policy, audit result, seed,
or derived artifact changes the ID.

## Compatibility

The global builder writes the authoritative manifest beside the dataset and an
absolute-path compatibility copy under the run directory. Both have the same
dataset ID. Training auto-discovers one shared adjacent manifest or accepts an
explicit `dataset_manifest` configuration value. Missing manifests are recorded
as `legacy_unverified`; an explicit or discovered invalid manifest is fatal.

Evaluators record the validated dataset ID when `--manifest` is supplied. Results
without it remain legacy/unverified and cannot support corrected headline claims.

Fixed and multi-packed datasets retain the compressed NPZ as the complete packing
record and provide manifest-tracked `uint8` `*_X.npy` and `*_Y.npy` sidecars.
Training with `use_mmap: true` reads these sidecars without materializing the full
dataset in RAM. Dynamic datasets provide `*_X.npy` and `*_lengths.npy` instead.

Leakage audits retain the protein-cluster assignments and nucleotide/protein
nearest-neighbor results as hashed artifacts. Reconstructable FASTA inputs and
aligner databases are removed after a successful audit.

## Corrected Dataset Freeze

The first corrected training program uses the pinned 24-genome inventory in
`configs/corrected_codonlm_dataset_v1.yaml`. Verify that every local GBFF still
matches its declared assembly ID, byte size, and SHA-256 digest without creating
derived data:

```bash
python -m scripts.freeze_corrected_datasets \
  --config configs/corrected_codonlm_dataset_v1.yaml \
  --freeze-id corrected-codonlm-v1 \
  --verify-sources-only
```

Build both genome- and genus-held-out protocols and bind their manifests into one
content-addressed `freeze.json`:

```bash
python -m scripts.freeze_corrected_datasets \
  --config configs/corrected_codonlm_dataset_v1.yaml \
  --freeze-id corrected-codonlm-v1 \
  --seed 1337 \
  --audit-threads 4
```

MMseqs2 is mandatory for this command, and the freeze refuses to overwrite an
existing freeze ID. The corrected grouped protocols quarantine exact cross-split CDS
copies and report all protein-cluster crossings and nearest-neighbor identities.
Protein homology is diagnostic here because homologous genes are expected across
whole genomes and genera; strict blocking remains mandatory for explicitly
homology-held-out evaluations.

Freeze schema v2 computes its identity without storage paths, so relocating a
byte-identical build does not change the aggregate freeze ID. To migrate or restore
the index for already completed protocol artifacts without repeating the expensive
alignment audits:

```bash
python -m scripts.freeze_corrected_datasets \
  --config configs/corrected_codonlm_dataset_v1.yaml \
  --freeze-id corrected-codonlm-v1 \
  --seed 1337 \
  --finalize-existing
```

The reviewed release contract is
`configs/corrected_codonlm_freeze_v1.json`. Verify the complete local artifact tree,
including every source, manifest, packed array, audit, vocabulary, and the approved
dataset identities, before training:

```bash
python -m scripts.verify_dataset_freeze \
  --freeze data/processed/corrected/corrected-codonlm-v1/freeze.json \
  --expected configs/corrected_codonlm_freeze_v1.json
```

The first portable aggregate freeze ID is
`1582505ae40445422711fa15918ee9c229caf84b1b3feba1a71f078259892249`.
The underlying genome and genus dataset IDs are unchanged by the index migration.
