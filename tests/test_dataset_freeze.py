from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from scripts.freeze_corrected_datasets import (
    build_freeze_index,
    load_and_validate_source_config,
    validate_protocol_manifests,
)


def _source_config(tmp_path: Path) -> Path:
    datasets = []
    for index in range(3):
        source = tmp_path / f"source-{index}.gbff"
        source.write_bytes(f"source-{index}".encode())
        datasets.append(
            {
                "name": f"source_{index}",
                "genome_id": f"GCF_{index:09d}.1",
                "gbff": str(source),
                "bytes": source.stat().st_size,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        )
    path = tmp_path / "sources.yaml"
    path.write_text(yaml.safe_dump({"datasets": datasets}))
    return path


def _protocol_manifest(protocol: str) -> dict:
    return {
        "dataset": {"id": f"{protocol}-id", "scientific_valid": True},
        "split_policy": {
            "effective_group_by": protocol,
            "record_counts": {"train": 8, "val": 1, "test": 1},
            "group_counts": {"train": 8, "val": 1, "test": 1},
            "achieved_record_fractions": {"train": 0.8, "val": 0.1, "test": 0.1},
        },
        "leakage_audit": {"status": "passed"},
        "sources": {
            "GCF_000000001.1": {"sha256": "a" * 64, "bytes": 100},
            "GCF_000000002.1": {"sha256": "b" * 64, "bytes": 200},
        },
        "vocabulary": {
            "sha256": "c" * 64,
            "size": 68,
            "special_tokens": {
                "<PAD>": 0,
                "<BOS_CDS>": 1,
                "<EOS_CDS>": 2,
                "<SEP>": 3,
            },
        },
        "packing": {
            "schema_version": 1,
            "mode": "multi",
            "block_size": 512,
            "transition_policy": "exactly_once",
        },
    }


def test_source_inventory_requires_and_verifies_content_pins(tmp_path):
    config_path = _source_config(tmp_path)
    config = load_and_validate_source_config(config_path)
    assert len(config["datasets"]) == 3

    Path(config["datasets"][0]["gbff"]).write_text("drifted")
    with pytest.raises(ValueError, match="Source size mismatch|Source SHA-256 mismatch"):
        load_and_validate_source_config(config_path)


def test_source_inventory_rejects_duplicate_genome_ids(tmp_path):
    config_path = _source_config(tmp_path)
    config = yaml.safe_load(config_path.read_text())
    config["datasets"][1]["genome_id"] = config["datasets"][0]["genome_id"]
    config_path.write_text(yaml.safe_dump(config))

    with pytest.raises(ValueError, match="duplicate genome_id"):
        load_and_validate_source_config(config_path)


def test_freeze_index_binds_both_scientific_protocols(tmp_path):
    config_path = _source_config(tmp_path)
    manifests = {}
    for protocol in ("genome", "genus"):
        path = tmp_path / f"{protocol}-manifest.json"
        path.write_text(json.dumps(_protocol_manifest(protocol), sort_keys=True))
        manifests[protocol] = (_protocol_manifest(protocol), path)

    index = build_freeze_index(
        config_path=config_path,
        seed=1337,
        manifests=manifests,
    )

    assert len(index["freeze_id"]) == 64
    assert index["split_seed"] == 1337
    assert set(index["protocols"]) == {"genome", "genus"}
    assert index["protocols"]["genome"]["dataset_id"] == "genome-id"


def test_freeze_rejects_cross_protocol_source_drift(tmp_path):
    genome = _protocol_manifest("genome")
    genus = _protocol_manifest("genus")
    genus["sources"]["GCF_000000002.1"]["bytes"] = 201

    with pytest.raises(ValueError, match="same sources"):
        validate_protocol_manifests(
            {
                "genome": (genome, tmp_path / "genome.json"),
                "genus": (genus, tmp_path / "genus.json"),
            }
        )


def test_freeze_rejects_non_scientific_manifest(tmp_path):
    genome = _protocol_manifest("genome")
    genome["dataset"]["scientific_valid"] = False

    with pytest.raises(ValueError, match="not marked scientific_valid"):
        validate_protocol_manifests(
            {
                "genome": (genome, tmp_path / "genome.json"),
                "genus": (_protocol_manifest("genus"), tmp_path / "genus.json"),
            }
        )
