"""Fail-closed dataset and checkpoint provenance for corrected evaluations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from .dataset_manifest import file_sha256, load_dataset_manifest, manifest_artifact_path


class EvaluationProvenanceError(ValueError):
    """Raised when evaluation inputs cannot be bound to one frozen dataset."""


def artifact_provenance(path: str | Path) -> dict:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise EvaluationProvenanceError(f"evaluation artifact not found: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def bind_dataset_manifest(
    manifest_path: str | Path,
    *,
    expected_artifacts: Mapping[str, str | Path] | None = None,
    require_scientific: bool = True,
) -> tuple[dict, dict]:
    resolved = Path(manifest_path).expanduser().resolve()
    manifest = load_dataset_manifest(resolved)
    if require_scientific and not manifest["dataset"].get("scientific_valid"):
        raise EvaluationProvenanceError(
            f"dataset manifest is not marked scientific_valid: {resolved}"
        )

    bound_artifacts = {}
    for name, selected_path in (expected_artifacts or {}).items():
        selected = Path(selected_path).expanduser().resolve()
        declared = manifest_artifact_path(manifest, resolved, name).resolve()
        if selected != declared:
            raise EvaluationProvenanceError(
                f"{name} input {selected} does not match manifest artifact {declared}"
            )
        bound_artifacts[name] = artifact_provenance(declared)

    vocabulary_path = manifest_artifact_path(manifest, resolved, "vocabulary").resolve()

    provenance = {
        "status": "frozen_manifest_verified",
        **artifact_provenance(resolved),
        "dataset_id": manifest["dataset"]["id"],
        "scientific_valid": bool(manifest["dataset"]["scientific_valid"]),
        "schema": manifest["schema"],
        "vocabulary": artifact_provenance(vocabulary_path),
        "bound_artifacts": bound_artifacts,
    }
    return manifest, provenance


def bind_checkpoint_dataset(
    checkpoint_cfg: Mapping,
    manifest_provenance: Mapping | None,
) -> dict:
    checkpoint_manifest = checkpoint_cfg.get("dataset_manifest")
    checkpoint_dataset_id = (
        checkpoint_manifest.get("dataset_id")
        if isinstance(checkpoint_manifest, Mapping)
        else None
    )
    if checkpoint_dataset_id is None:
        return {
            "status": "legacy_checkpoint_unverified",
            "dataset_id": None,
        }
    if manifest_provenance is None:
        raise EvaluationProvenanceError(
            "corrected checkpoint requires an explicit frozen dataset manifest"
        )
    selected_dataset_id = manifest_provenance.get("dataset_id")
    if checkpoint_dataset_id != selected_dataset_id:
        raise EvaluationProvenanceError(
            "checkpoint dataset identity mismatch: "
            f"checkpoint={checkpoint_dataset_id!r}, manifest={selected_dataset_id!r}"
        )
    checkpoint_vocabulary = checkpoint_cfg.get("vocabulary")
    checkpoint_vocabulary_sha = (
        checkpoint_vocabulary.get("sha256")
        if isinstance(checkpoint_vocabulary, Mapping)
        else None
    )
    manifest_vocabulary_sha = (
        manifest_provenance.get("vocabulary", {}).get("sha256")
    )
    if (
        checkpoint_vocabulary_sha is not None
        and checkpoint_vocabulary_sha != manifest_vocabulary_sha
    ):
        raise EvaluationProvenanceError(
            "checkpoint vocabulary mismatch: "
            f"checkpoint={checkpoint_vocabulary_sha!r}, manifest={manifest_vocabulary_sha!r}"
        )
    return {
        "status": "checkpoint_manifest_verified",
        "dataset_id": checkpoint_dataset_id,
        "vocabulary_sha256": checkpoint_vocabulary_sha,
    }


def bind_derived_dataset(
    artifact_path: str | Path,
    provenance_path: str | Path,
    *,
    manifest_provenance: Mapping,
    source_artifact_path: str | Path,
) -> dict:
    """Verify a derived evaluator input against its frozen source artifact."""
    resolved_provenance = Path(provenance_path).expanduser().resolve()
    try:
        provenance = json.loads(resolved_provenance.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationProvenanceError(
            f"cannot read derived dataset provenance {resolved_provenance}: {exc}"
        ) from exc

    if provenance.get("status") != "derived_control_verified":
        raise EvaluationProvenanceError(
            f"unsupported derived dataset provenance status: {provenance.get('status')!r}"
        )
    if provenance.get("dataset_id") != manifest_provenance.get("dataset_id"):
        raise EvaluationProvenanceError("derived dataset manifest identity mismatch")

    expected_vocabulary = manifest_provenance.get("vocabulary", {}).get("sha256")
    if provenance.get("vocabulary", {}).get("sha256") != expected_vocabulary:
        raise EvaluationProvenanceError("derived dataset vocabulary mismatch")

    comparisons = (
        ("derived output", artifact_provenance(artifact_path), provenance.get("output")),
        (
            "derived source",
            artifact_provenance(source_artifact_path),
            provenance.get("source_test"),
        ),
    )
    for label, current, declared in comparisons:
        if not isinstance(declared, Mapping):
            raise EvaluationProvenanceError(f"{label} provenance is missing")
        if any(current.get(key) != declared.get(key) for key in ("path", "bytes", "sha256")):
            raise EvaluationProvenanceError(f"{label} provenance mismatch")

    return {
        "status": "derived_dataset_verified",
        "provenance": artifact_provenance(resolved_provenance),
        "derivation": provenance,
    }


def bind_embedding_artifact(path: str | Path, *, require_verified: bool) -> dict:
    embedding = Path(path).expanduser().resolve()
    metadata_path = embedding.with_suffix(embedding.suffix + ".metadata.json")
    if not metadata_path.is_file():
        if require_verified:
            raise EvaluationProvenanceError(
                f"verified embedding metadata sidecar not found: {metadata_path}"
            )
        return {
            "status": "legacy_embedding_unverified",
            "embedding": artifact_provenance(embedding),
        }
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationProvenanceError(
            f"cannot read embedding metadata {metadata_path}: {exc}"
        ) from exc
    if require_verified:
        if metadata.get("validation_status") != "causal_verified":
            raise EvaluationProvenanceError(
                f"embedding is not causally verified: {embedding}"
            )
        if metadata.get("dataset_manifest", {}).get("status") != "frozen_manifest_verified":
            raise EvaluationProvenanceError(
                f"embedding lacks frozen dataset provenance: {embedding}"
            )
        if metadata.get("checkpoint_dataset", {}).get("status") != "checkpoint_manifest_verified":
            raise EvaluationProvenanceError(
                f"embedding checkpoint is not bound to its dataset: {embedding}"
            )
    return {
        "status": "verified_embedding" if require_verified else "metadata_present",
        "embedding": artifact_provenance(embedding),
        "metadata": artifact_provenance(metadata_path),
        "extraction": metadata,
    }


def bind_embedding_pair(
    train_path: str | Path,
    test_path: str | Path,
    *,
    require_verified: bool,
) -> dict:
    train = bind_embedding_artifact(train_path, require_verified=require_verified)
    test = bind_embedding_artifact(test_path, require_verified=require_verified)
    if require_verified:
        train_extraction = train["extraction"]
        test_extraction = test["extraction"]
        comparisons = {
            "dataset_id": (
                train_extraction["dataset_manifest"].get("dataset_id"),
                test_extraction["dataset_manifest"].get("dataset_id"),
            ),
            "checkpoint_sha256": (
                train_extraction["checkpoint"].get("sha256"),
                test_extraction["checkpoint"].get("sha256"),
            ),
            "vocabulary_sha256": (
                train_extraction["vocabulary"].get("sha256"),
                test_extraction["vocabulary"].get("sha256"),
            ),
        }
        mismatches = {
            name: values for name, values in comparisons.items() if values[0] != values[1]
        }
        if mismatches:
            raise EvaluationProvenanceError(
                f"train/test embedding provenance mismatch: {mismatches}"
            )
    return {"train": train, "test": test}


__all__ = [
    "EvaluationProvenanceError",
    "artifact_provenance",
    "bind_embedding_artifact",
    "bind_embedding_pair",
    "bind_checkpoint_dataset",
    "bind_derived_dataset",
    "bind_dataset_manifest",
]
