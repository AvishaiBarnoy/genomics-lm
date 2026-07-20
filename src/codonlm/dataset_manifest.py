"""Versioned, content-addressed dataset manifest contracts."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .training.vocabulary import dataset_token_bounds, load_itos

SCHEMA_NAME = "codonlm_dataset_manifest"
SCHEMA_VERSION = 1
SPLITS = ("train", "val", "test")


class DatasetManifestError(ValueError):
    """Raised when a dataset manifest is unsupported or inconsistent."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_entry(path: Path, manifest_dir: Path, role: str) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        stored_path = str(resolved.relative_to(manifest_dir.resolve()))
    except ValueError:
        stored_path = str(resolved)
    return {
        "path": stored_path,
        "role": role,
        "bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def _identity_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(manifest)
    payload.get("dataset", {}).pop("id", None)
    for compatibility_key in ("train", "val", "test", "datasets", "genome_sources"):
        payload.pop(compatibility_key, None)
    payload.get("vocabulary", {}).pop("itos_path", None)
    for artifact in payload.get("artifacts", {}).values():
        artifact.pop("path", None)
    for source in payload.get("sources", {}).values():
        source.pop("path", None)
    return payload


def dataset_identity(manifest: dict[str, Any]) -> str:
    encoded = json.dumps(
        _identity_payload(manifest), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def finalize_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(manifest)
    result.setdefault("dataset", {})["id"] = dataset_identity(result)
    return result


def _require(mapping: dict, key: str, context: str):
    if key not in mapping:
        raise DatasetManifestError(f"missing {context}.{key}")
    return mapping[key]


def _resolve_artifact(manifest_path: Path, entry: dict) -> Path:
    path = Path(_require(entry, "path", "artifact"))
    return path if path.is_absolute() else manifest_path.parent / path


def manifest_artifact_path(manifest: dict, manifest_path: Path, name: str) -> Path:
    return _resolve_artifact(manifest_path, _require(manifest["artifacts"], name, "artifacts"))


def validate_dataset_manifest(
    manifest: dict[str, Any], manifest_path: Path, *, verify_artifacts: bool = True
) -> dict[str, Any]:
    schema = _require(manifest, "schema", "manifest")
    if schema.get("name") != SCHEMA_NAME or schema.get("version") != SCHEMA_VERSION:
        raise DatasetManifestError(
            f"unsupported dataset manifest schema: {schema!r}; expected {SCHEMA_NAME} v{SCHEMA_VERSION}"
        )
    dataset = _require(manifest, "dataset", "manifest")
    declared_id = _require(dataset, "id", "dataset")
    computed_id = dataset_identity(manifest)
    if declared_id != computed_id:
        raise DatasetManifestError(
            f"dataset identity mismatch: declared={declared_id}, computed={computed_id}"
        )
    split_policy = _require(manifest, "split_policy", "manifest")
    counts = _require(split_policy, "record_counts", "split_policy")
    if set(counts) != set(SPLITS) or any(int(counts[name]) < 0 for name in SPLITS):
        raise DatasetManifestError("split record_counts must contain non-negative train/val/test")
    if sum(int(counts[name]) for name in SPLITS) != int(dataset["source_record_count"]):
        raise DatasetManifestError("split record counts do not sum to dataset source_record_count")
    requested = _require(split_policy, "requested_fractions", "split_policy")
    if any(not 0.0 <= float(value) < 1.0 for value in requested.values()):
        raise DatasetManifestError("requested split fractions must be in [0, 1)")
    groups = split_policy.get("groups_by_split")
    if groups:
        group_sets = [set(groups[name]) for name in SPLITS]
        if any(group_sets[i] & group_sets[j] for i in range(3) for j in range(i + 1, 3)):
            raise DatasetManifestError("split groups overlap")
    scientific = bool(dataset.get("scientific_valid"))
    if scientific != bool(split_policy.get("scientific_valid")):
        raise DatasetManifestError("dataset and split_policy scientific_valid flags disagree")
    leakage = _require(manifest, "leakage_audit", "manifest")
    if scientific and (
        split_policy.get("effective_group_by") == "sequence"
        or split_policy.get("allow_sequence_split")
        or leakage.get("status") != "passed"
        or leakage.get("homology_audit_skipped")
        or leakage.get("exact_duplicate_override")
    ):
        raise DatasetManifestError("unsafe preparation cannot be marked scientific_valid")
    vocabulary = _require(manifest, "vocabulary", "manifest")
    sources = _require(manifest, "sources", "manifest")
    tokenization = _require(manifest, "tokenization", "manifest")
    packing = _require(manifest, "packing", "manifest")
    reproducibility = _require(manifest, "reproducibility", "manifest")
    _require(tokenization, "ambiguous_codon_policy", "tokenization")
    if packing.get("mode") not in {"fixed", "dynamic", "multi"}:
        raise DatasetManifestError("packing.mode must be fixed, dynamic, or multi")
    if packing.get("transition_policy") != "exactly_once":
        raise DatasetManifestError("packing transition_policy must be exactly_once")
    for seed_name in ("split_seed", "packing_seed"):
        _require(reproducibility, seed_name, "reproducibility")
    for token_name in ("<PAD>", "<BOS_CDS>", "<EOS_CDS>", "<SEP>"):
        _require(vocabulary.get("special_tokens", {}), token_name, "vocabulary.special_tokens")
    artifacts = _require(manifest, "artifacts", "manifest")
    for required in (
        "train_tokens", "val_tokens", "test_tokens", "vocabulary",
        "source_metadata", "source_dna", "fragment_metadata", "leakage_audit",
        "train_packing_metadata", "val_packing_metadata", "test_packing_metadata",
    ):
        _require(artifacts, required, "artifacts")
    if verify_artifacts:
        for source_name, source in sources.items():
            source_path = Path(source["path"])
            if not source_path.exists():
                raise DatasetManifestError(f"source {source_name} not found: {source_path}")
            if source_path.stat().st_size != int(source["bytes"]):
                raise DatasetManifestError(f"source {source_name} size mismatch")
            if file_sha256(source_path) != source["sha256"]:
                raise DatasetManifestError(f"source {source_name} hash mismatch")
        for name, entry in artifacts.items():
            path = _resolve_artifact(manifest_path, entry)
            if not path.exists():
                raise DatasetManifestError(f"artifact {name} not found: {path}")
            if path.stat().st_size != int(entry["bytes"]):
                raise DatasetManifestError(f"artifact {name} size mismatch: {path}")
            if file_sha256(path) != entry["sha256"]:
                raise DatasetManifestError(f"artifact {name} hash mismatch: {path}")
        vocab_path = _resolve_artifact(manifest_path, artifacts["vocabulary"])
        tokens = load_itos(vocab_path)
        if len(tokens) != int(vocabulary["size"]):
            raise DatasetManifestError("vocabulary size does not match artifact")
        if file_sha256(vocab_path) != vocabulary["sha256"]:
            raise DatasetManifestError("vocabulary hash does not match artifact")
        for token_name, token_id in vocabulary["special_tokens"].items():
            if int(token_id) < 0 or int(token_id) >= len(tokens) or tokens[int(token_id)] != token_name:
                raise DatasetManifestError(f"special token mapping is invalid for {token_name}")
        for split in SPLITS:
            data_path = _resolve_artifact(manifest_path, artifacts[f"{split}_tokens"])
            for suffix, role_suffix in (
                ("_X.npy", "x_npy"), ("_Y.npy", "y_npy"),
                ("_lengths.npy", "lengths_npy"),
            ):
                sidecar = data_path.with_name(data_path.stem + suffix)
                if sidecar.exists() and f"{split}_{role_suffix}" not in artifacts:
                    raise DatasetManifestError(
                        f"untracked memory-map sidecar for {split}: {sidecar}"
                    )
            bounds = dataset_token_bounds(data_path)
            if bounds.minimum is not None and bounds.minimum < 0:
                raise DatasetManifestError(f"{split} contains negative token IDs")
            if bounds.maximum is not None and bounds.maximum >= len(tokens):
                raise DatasetManifestError(f"{split} token IDs exceed vocabulary")
    return manifest


def load_dataset_manifest(path: str | Path, *, verify_artifacts: bool = True):
    manifest_path = Path(path).expanduser().resolve()
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetManifestError(f"cannot load dataset manifest {manifest_path}: {exc}") from exc
    validate_dataset_manifest(manifest, manifest_path, verify_artifacts=verify_artifacts)
    return manifest


def discover_manifest(dataset_paths: Iterable[str | Path]) -> Path | None:
    candidates = {Path(path).expanduser().resolve().parent / "manifest.json" for path in dataset_paths}
    existing = {path for path in candidates if path.exists()}
    if not existing:
        return None
    if len(existing) != 1 or len(candidates) != 1:
        raise DatasetManifestError("dataset shards do not share one adjacent manifest.json")
    return existing.pop()


__all__ = [
    "DatasetManifestError", "SCHEMA_NAME", "SCHEMA_VERSION", "artifact_entry",
    "dataset_identity", "discover_manifest", "file_sha256", "finalize_manifest",
    "load_dataset_manifest", "manifest_artifact_path", "validate_dataset_manifest",
]
