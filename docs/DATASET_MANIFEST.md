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
