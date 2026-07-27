import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from scripts.training_preflight import _write_fixture
from scripts.audit_generated_sequences import _read_manifest_training
from src.codonlm.dataset_manifest import artifact_entry, finalize_manifest
from src.codonlm.evaluation_provenance import (
    EvaluationProvenanceError,
    artifact_provenance,
    bind_checkpoint_dataset,
    bind_dataset_manifest,
)
from src.codonlm.model_tiny_gpt import TinyGPT


def _corrected_run(tmp_path: Path):
    manifest_path, manifest = _write_fixture(tmp_path / "freeze")
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True)
    vocabulary_path = manifest_path.parent / "itos.txt"
    (run_dir / "itos.txt").write_bytes(vocabulary_path.read_bytes())
    cfg = {
        "vocab_size": 8,
        "block_size": 8,
        "n_layer": 1,
        "n_head": 1,
        "n_embd": 16,
        "dropout": 0.0,
        "use_sdpa": True,
        "val_npz": [str(manifest_path.parent / "val.npz")],
        "test_npz": [str(manifest_path.parent / "test.npz")],
        "dataset_manifest": {
            "dataset_id": manifest["dataset"]["id"],
            "path": str(manifest_path),
        },
        "vocabulary": {"sha256": manifest["vocabulary"]["sha256"]},
    }
    model = TinyGPT(
        vocab_size=8,
        block_size=8,
        n_layer=1,
        n_head=1,
        n_embd=16,
        dropout=0.0,
        use_sdpa=True,
    )
    torch.save(
        {"model": model.state_dict(), "cfg": cfg},
        run_dir / "checkpoints" / "best.pt",
    )
    return manifest_path, manifest, run_dir


def test_manifest_binding_verifies_declared_artifacts(tmp_path):
    manifest_path, manifest = _write_fixture(tmp_path)
    _, provenance = bind_dataset_manifest(
        manifest_path,
        expected_artifacts={
            "train_tokens": manifest_path.parent / "train.npz",
            "test_tokens": manifest_path.parent / "test.npz",
        },
    )

    assert provenance["status"] == "frozen_manifest_verified"
    assert provenance["dataset_id"] == manifest["dataset"]["id"]
    assert set(provenance["bound_artifacts"]) == {"train_tokens", "test_tokens"}

    wrong = tmp_path / "wrong.npz"
    wrong.touch()
    with pytest.raises(EvaluationProvenanceError, match="does not match"):
        bind_dataset_manifest(
            manifest_path, expected_artifacts={"test_tokens": wrong}
        )


def test_corrected_checkpoint_requires_matching_manifest_and_vocabulary(tmp_path):
    manifest_path, manifest = _write_fixture(tmp_path)
    _, provenance = bind_dataset_manifest(manifest_path)
    cfg = {
        "dataset_manifest": {"dataset_id": manifest["dataset"]["id"]},
        "vocabulary": {"sha256": manifest["vocabulary"]["sha256"]},
    }
    assert bind_checkpoint_dataset(cfg, provenance)["status"] == "checkpoint_manifest_verified"

    with pytest.raises(EvaluationProvenanceError, match="explicit frozen"):
        bind_checkpoint_dataset(cfg, None)
    with pytest.raises(EvaluationProvenanceError, match="identity mismatch"):
        bind_checkpoint_dataset(
            {"dataset_manifest": {"dataset_id": "different"}}, provenance
        )
    with pytest.raises(EvaluationProvenanceError, match="vocabulary mismatch"):
        bind_checkpoint_dataset(
            {**cfg, "vocabulary": {"sha256": "different"}}, provenance
        )


def test_corrected_test_evaluation_emits_frozen_provenance(tmp_path):
    manifest_path, manifest, run_dir = _corrected_run(tmp_path)
    command = [
        "python",
        "-m",
        "scripts.evaluate_test",
        "--run_dir",
        str(run_dir),
        "--manifest",
        str(manifest_path),
        "--batch_size",
        "2",
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env={**os.environ, "FORCE_CPU": "1"},
    )
    assert result.returncode == 0, result.stderr

    report = json.loads((run_dir / "scores" / "metrics.json").read_text())
    provenance = report["test_evaluation_provenance"]
    assert provenance["dataset_manifest"]["dataset_id"] == manifest["dataset"]["id"]
    assert provenance["checkpoint_dataset"]["status"] == "checkpoint_manifest_verified"
    assert report["test_evaluated_tokens"] > 0
    assert report["test_loss"] == report["test_nll"]
    assert report["test_ppl"] == pytest.approx(np.exp(report["test_nll"]))
    assert provenance["loss_definition"]["nll"] == "unsmoothed_cross_entropy"

    without_manifest = subprocess.run(
        command[:5] + command[7:],
        capture_output=True,
        text=True,
        env={**os.environ, "FORCE_CPU": "1"},
    )
    assert without_manifest.returncode != 0
    assert "explicit frozen dataset manifest" in without_manifest.stderr


def test_corrected_test_evaluation_namespaces_checkpoint_metrics(tmp_path):
    manifest_path, _, run_dir = _corrected_run(tmp_path)
    best = torch.load(run_dir / "checkpoints" / "best.pt", map_location="cpu")
    torch.save(best, run_dir / "checkpoints" / "last.pt")
    command = [
        "python",
        "-m",
        "scripts.evaluate_test",
        "--run_dir",
        str(run_dir),
        "--manifest",
        str(manifest_path),
        "--checkpoint-name",
        "last.pt",
        "--metric-prefix",
        "last_test",
        "--batch_size",
        "2",
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env={**os.environ, "FORCE_CPU": "1"},
    )
    assert result.returncode == 0, result.stderr

    report = json.loads((run_dir / "scores" / "metrics.json").read_text())
    assert report["last_test_ppl"] == pytest.approx(np.exp(report["last_test_nll"]))
    checkpoint = report["last_test_evaluation_provenance"]["checkpoint"]
    assert checkpoint["path"].endswith("last.pt")


def test_corrected_validation_evaluation_binds_validation_artifact(tmp_path):
    manifest_path, manifest, run_dir = _corrected_run(tmp_path)
    command = [
        "python",
        "-m",
        "scripts.evaluate_test",
        "--run_dir",
        str(run_dir),
        "--split",
        "validation",
        "--manifest",
        str(manifest_path),
        "--batch_size",
        "2",
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env={**os.environ, "FORCE_CPU": "1"},
    )
    assert result.returncode == 0, result.stderr

    report = json.loads((run_dir / "scores" / "metrics.json").read_text())
    assert report["validation_ppl"] == pytest.approx(
        np.exp(report["validation_nll"])
    )
    provenance = report["validation_evaluation_provenance"]
    assert provenance["dataset_manifest"]["dataset_id"] == manifest["dataset"]["id"]
    assert set(provenance["dataset_manifest"]["bound_artifacts"]) == {"val_tokens"}
    assert provenance["val_tokens"]["path"].endswith("val.npz")
    assert "test_tokens" not in provenance


def test_corrected_test_evaluation_accepts_verified_derived_control(tmp_path):
    manifest_path, manifest, run_dir = _corrected_run(tmp_path)
    source = manifest_path.parent / "test.npz"
    control = tmp_path / "control.npz"
    control.write_bytes(source.read_bytes())
    sidecar = control.with_suffix(".npz.provenance.json")
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "derived_control_verified",
                "control": "fixture",
                "seed": 1337,
                "dataset_id": manifest["dataset"]["id"],
                "vocabulary": artifact_provenance(manifest_path.parent / "itos.txt"),
                "source_test": artifact_provenance(source),
                "output": artifact_provenance(control),
            }
        )
    )
    command = [
        "python",
        "-m",
        "scripts.evaluate_test",
        "--run_dir",
        str(run_dir),
        "--test_npz",
        str(control),
        "--manifest",
        str(manifest_path),
        "--derived_provenance",
        str(sidecar),
        "--batch_size",
        "2",
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env={**os.environ, "FORCE_CPU": "1"},
    )
    assert result.returncode == 0, result.stderr
    report = json.loads((run_dir / "scores" / "metrics.json").read_text())
    assert (
        report["test_evaluation_provenance"]["derived_dataset"]["status"]
        == "derived_dataset_verified"
    )

    control.write_bytes(control.read_bytes() + b"tampered")
    failed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env={**os.environ, "FORCE_CPU": "1"},
    )
    assert failed.returncode != 0
    assert "derived output provenance mismatch" in failed.stderr


def test_corrected_embedding_extraction_records_dataset_identity(tmp_path):
    manifest_path, manifest, run_dir = _corrected_run(tmp_path)
    fasta = tmp_path / "input.fasta"
    fasta.write_text(">gene-1\nAAACCC\n")
    output = tmp_path / "embeddings.npz"
    command = [
        "python",
        "-m",
        "scripts.extract_embeddings",
        "--run_dir",
        str(run_dir),
        "--fasta",
        str(fasta),
        "--manifest",
        str(manifest_path),
        "--out",
        str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    metadata = json.loads(output.with_suffix(".npz.metadata.json").read_text())
    assert metadata["dataset_manifest"]["dataset_id"] == manifest["dataset"]["id"]
    assert metadata["checkpoint_dataset"]["status"] == "checkpoint_manifest_verified"


def _verified_embeddings(path: Path, ids: list[str], dataset_id: str):
    values = np.array(
        [[0.0, 0.1], [0.2, 0.0], [0.1, 0.2], [1.0, 0.9], [0.8, 1.0], [0.9, 0.8]],
        dtype=np.float32,
    )
    np.savez(path, X=values, ids=np.asarray(ids, dtype=object))
    metadata = {
        "validation_status": "causal_verified",
        "dataset_manifest": {
            "status": "frozen_manifest_verified",
            "dataset_id": dataset_id,
        },
        "checkpoint_dataset": {
            "status": "checkpoint_manifest_verified",
            "dataset_id": dataset_id,
        },
        "checkpoint": {"sha256": "checkpoint-sha"},
        "vocabulary": {"sha256": "vocabulary-sha"},
    }
    path.with_suffix(".npz.metadata.json").write_text(json.dumps(metadata))


def test_classifier_requires_matching_verified_embedding_provenance(tmp_path):
    ids = [f"gene-{index}" for index in range(6)]
    train_embeddings = tmp_path / "train.npz"
    test_embeddings = tmp_path / "test.npz"
    _verified_embeddings(train_embeddings, ids, "dataset-a")
    _verified_embeddings(test_embeddings, ids, "dataset-a")
    for split in ("train", "test"):
        labels = tmp_path / f"{split}_labels.csv"
        labels.write_text(
            "id,label\n"
            + "".join(f"{identifier},{0 if index < 3 else 1}\n" for index, identifier in enumerate(ids))
        )
    output = tmp_path / "report"
    config = tmp_path / "probe.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "task": "fixture",
                "protocol": "corrected",
                "out_dir": str(output),
                "require_verified_embeddings": True,
                "data": {
                    "train_embeddings": str(train_embeddings),
                    "test_embeddings": str(test_embeddings),
                    "train_labels": str(tmp_path / "train_labels.csv"),
                    "test_labels": str(tmp_path / "test_labels.csv"),
                },
                "classifier": {"kind": "probe_logreg", "C": 1.0},
            }
        )
    )

    result = subprocess.run(
        ["python", "-m", "scripts.train_classifier", "--config", str(config)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["evaluation_provenance"]["status"] == "verified"
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["embedding_inputs"]["train"]["status"] == "verified_embedding"

    test_metadata = test_embeddings.with_suffix(".npz.metadata.json")
    mismatched = json.loads(test_metadata.read_text())
    mismatched["checkpoint"]["sha256"] = "different"
    test_metadata.write_text(json.dumps(mismatched))
    failed = subprocess.run(
        ["python", "-m", "scripts.train_classifier", "--config", str(config)],
        capture_output=True,
        text=True,
    )
    assert failed.returncode != 0
    assert "embedding provenance mismatch" in failed.stderr


def test_generated_audit_derives_train_only_records_from_manifest(tmp_path):
    manifest_path, manifest = _write_fixture(tmp_path)
    metadata_path = manifest_path.parent / "cds_meta.tsv"
    dna_path = manifest_path.parent / "cds_dna.txt"
    splits = ["train"] * 5 + ["val"] * 2 + ["test"] * 2
    metadata_path.write_text(
        "line_idx\tsplit\tsource_id\n"
        + "".join(
            f"{index}\t{split}\trecord-{index}\n"
            for index, split in enumerate(splits)
        )
    )
    dna_path.write_text("".join("AAACCCGGG\n" for _ in splits))
    manifest["artifacts"]["source_metadata"] = artifact_entry(
        metadata_path, manifest_path.parent, "source_metadata"
    )
    manifest["artifacts"]["source_dna"] = artifact_entry(
        dna_path, manifest_path.parent, "source_dna"
    )
    manifest = finalize_manifest(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    records, provenance = _read_manifest_training(manifest_path)

    assert [record["source_id"] for record in records] == [
        f"record-{index}" for index in range(5)
    ]
    assert provenance["training_source"]["record_count"] == 5
    assert provenance["training_source"]["selection"] == "source_metadata.split == train"
