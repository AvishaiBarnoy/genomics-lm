import copy
import json
import shutil

import numpy as np
import pytest

from src.codonlm.dataset_manifest import (
    DatasetManifestError,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    artifact_entry,
    dataset_identity,
    finalize_manifest,
    load_dataset_manifest,
    validate_dataset_manifest,
)


def _fixture(tmp_path, mode="fixed", mmap_train=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "source.gbff"
    source.write_text("source snapshot\n")
    vocab = tmp_path / "itos.txt"
    vocab.write_text("<PAD>\n<BOS_CDS>\n<EOS_CDS>\n<SEP>\nAAA\n")
    required_files = {
        "vocabulary": vocab,
        "source_metadata": tmp_path / "cds_meta.tsv",
        "source_dna": tmp_path / "cds_dna.txt",
        "fragment_metadata": tmp_path / "cds_fragments.tsv",
        "leakage_audit": tmp_path / "leakage_audit.json",
    }
    for name, path in required_files.items():
        if name != "vocabulary":
            path.write_text(name + "\n")
    artifacts = {
        name: artifact_entry(path, tmp_path, name) for name, path in required_files.items()
    }
    for split in ("train", "val", "test"):
        path = tmp_path / f"{split}.npz"
        if mode == "dynamic":
            np.savez(path, X=np.array([1, 4, 2], dtype=np.int32), lengths=np.array([3]))
        else:
            np.savez(
                path,
                X=np.array([[1, 4]], dtype=np.int32),
                Y=np.array([[4, 2]], dtype=np.int32),
            )
        if split == "train" and mmap_train:
            path.write_bytes(b"")
            np.save(tmp_path / "train_X.npy", np.array([[1, 4]], dtype=np.int32))
            np.save(tmp_path / "train_Y.npy", np.array([[4, 2]], dtype=np.int32))
        artifacts[f"{split}_tokens"] = artifact_entry(path, tmp_path, f"{split}_tokens")
        packing = tmp_path / f"{split}_packing.tsv"
        packing.write_text("window_index\n0\n")
        artifacts[f"{split}_packing_metadata"] = artifact_entry(
            packing, tmp_path, f"{split}_packing_metadata"
        )
    if mmap_train:
        artifacts["train_x_npy"] = artifact_entry(tmp_path / "train_X.npy", tmp_path, "train_x_npy")
        artifacts["train_y_npy"] = artifact_entry(tmp_path / "train_Y.npy", tmp_path, "train_y_npy")
    manifest = {
        "schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
        "dataset": {"id": "pending", "scientific_valid": True, "source_record_count": 3},
        "sources": {
            "genome-1": {
                "path": str(source.resolve()),
                "sha256": artifacts["source_metadata"]["sha256"],
                "bytes": source.stat().st_size,
                "identity_source": "fixture",
            }
        },
        "split_policy": {
            "scientific_valid": True,
            "effective_group_by": "genome",
            "allow_sequence_split": False,
            "requested_fractions": {"val": 0.2, "test": 0.2},
            "record_counts": {"train": 1, "val": 1, "test": 1},
            "groups_by_split": {"train": ["a"], "val": ["b"], "test": ["c"]},
        },
        "vocabulary": {
            "sha256": artifacts["vocabulary"]["sha256"],
            "size": 5,
            "special_tokens": {"<PAD>": 0, "<BOS_CDS>": 1, "<EOS_CDS>": 2, "<SEP>": 3},
        },
        "leakage_audit": {
            "status": "passed",
            "homology_audit_skipped": False,
            "exact_duplicate_override": False,
        },
        "tokenization": {"ambiguous_codon_policy": {"name": "split"}},
        "packing": {"mode": mode, "transition_policy": "exactly_once"},
        "reproducibility": {"split_seed": 1337, "packing_seed": 1337},
        "artifacts": artifacts,
    }
    # Correct the intentionally independent source hash.
    from src.codonlm.dataset_manifest import file_sha256
    manifest["sources"]["genome-1"]["sha256"] = file_sha256(source)
    manifest = finalize_manifest(manifest)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest_path


@pytest.mark.parametrize("mode", ["fixed", "dynamic"])
def test_fixed_and_dynamic_manifests_validate(tmp_path, mode):
    path = _fixture(tmp_path / mode, mode=mode)
    assert load_dataset_manifest(path)["packing"]["mode"] == mode


def test_memmap_sidecars_must_be_in_inventory(tmp_path):
    path = _fixture(tmp_path, mmap_train=True)
    manifest = load_dataset_manifest(path)
    manifest["artifacts"].pop("train_x_npy")
    manifest = finalize_manifest(manifest)
    with pytest.raises(DatasetManifestError, match="untracked memory-map sidecar"):
        validate_dataset_manifest(manifest, path)


def test_modified_artifact_fails_hash_validation(tmp_path):
    path = _fixture(tmp_path)
    (tmp_path / "train.npz").write_bytes(b"changed")
    with pytest.raises(DatasetManifestError, match="size mismatch|hash mismatch"):
        load_dataset_manifest(path)


def test_unknown_schema_and_unsafe_scientific_claim_fail(tmp_path):
    path = _fixture(tmp_path)
    manifest = json.loads(path.read_text())
    manifest["schema"]["version"] = 999
    with pytest.raises(DatasetManifestError, match="unsupported"):
        validate_dataset_manifest(manifest, path, verify_artifacts=False)
    manifest = json.loads(path.read_text())
    manifest["split_policy"]["effective_group_by"] = "sequence"
    manifest = finalize_manifest(manifest)
    with pytest.raises(DatasetManifestError, match="unsafe preparation"):
        validate_dataset_manifest(manifest, path, verify_artifacts=False)


def test_identity_is_deterministic_and_relocation_independent(tmp_path):
    original_path = _fixture(tmp_path / "original")
    original = load_dataset_manifest(original_path)
    relocated_dir = tmp_path / "relocated"
    shutil.copytree(original_path.parent, relocated_dir)
    relocated_path = relocated_dir / "manifest.json"
    relocated = json.loads(relocated_path.read_text())
    relocated["sources"]["genome-1"]["path"] = str((relocated_dir / "source.gbff").resolve())
    relocated_path.write_text(json.dumps(relocated, indent=2, sort_keys=True))
    assert load_dataset_manifest(relocated_path)["dataset"]["id"] == original["dataset"]["id"]
    assert dataset_identity(copy.deepcopy(relocated)) == original["dataset"]["id"]
