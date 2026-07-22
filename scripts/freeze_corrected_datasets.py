#!/usr/bin/env python3
"""Build and bind corrected genome/genus holdout datasets into one freeze index."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from scripts.build_global_manifest import validate_pinned_source
from src.codonlm.dataset_manifest import file_sha256, load_dataset_manifest


FREEZE_SCHEMA_NAME = "codonlm_dataset_freeze"
FREEZE_SCHEMA_VERSION = 2
CONTRACT_SCHEMA_NAME = "codonlm_pipeline_freeze_contract"
CONTRACT_SCHEMA_VERSION = 1
GROUP_PROTOCOLS = ("genome", "genus")


def load_and_validate_source_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text()) or {}
    if not isinstance(cfg, dict):
        raise ValueError("dataset config must be a mapping")
    datasets = cfg.get("datasets")
    if not isinstance(datasets, list) or len(datasets) < 3:
        raise ValueError("dataset config must contain at least three pinned sources")

    genome_ids: set[str] = set()
    source_paths: set[Path] = set()
    for index, dataset in enumerate(datasets):
        if not isinstance(dataset, dict):
            raise ValueError(f"datasets[{index}] must be a mapping")
        genome_id = str(dataset.get("genome_id", "")).strip()
        if not genome_id:
            raise ValueError(f"datasets[{index}] is missing explicit genome_id")
        if genome_id in genome_ids:
            raise ValueError(f"duplicate genome_id in source inventory: {genome_id}")
        genome_ids.add(genome_id)

        if "sha256" not in dataset or "bytes" not in dataset:
            raise ValueError(f"source {genome_id} must pin sha256 and bytes")
        source_path = Path(str(dataset.get("gbff", ""))).expanduser().resolve()
        if source_path in source_paths:
            raise ValueError(f"duplicate source path in inventory: {source_path}")
        source_paths.add(source_path)
        if not source_path.is_file():
            raise ValueError(f"source {genome_id} not found: {source_path}")
        validate_pinned_source(dataset, source_path)
    return cfg


def validate_protocol_manifests(
    manifests: dict[str, tuple[dict[str, Any], Path]],
) -> None:
    reference_sources = None
    reference_vocab = None
    reference_packing = None
    reference_homology_policy = None
    for protocol in GROUP_PROTOCOLS:
        manifest, _ = manifests[protocol]
        if not manifest["dataset"].get("scientific_valid"):
            raise ValueError(f"{protocol} dataset is not marked scientific_valid")
        if manifest["split_policy"].get("effective_group_by") != protocol:
            raise ValueError(f"{protocol} manifest uses the wrong split protocol")
        if manifest["leakage_audit"].get("status") != "passed":
            raise ValueError(f"{protocol} leakage audit did not pass")
        if manifest["leakage_audit"].get("homology_audit_skipped"):
            raise ValueError(f"{protocol} homology audit was skipped")
        if manifest["leakage_audit"].get("exact_duplicate_override"):
            raise ValueError(f"{protocol} exact-duplicate audit was overridden")
        homology_policy = manifest["leakage_audit"].get(
            "protein_homology_policy", "block"
        )
        if homology_policy not in {"block", "report"}:
            raise ValueError(f"{protocol} manifest has an invalid homology policy")

        sources = {
            key: {"sha256": value["sha256"], "bytes": int(value["bytes"])}
            for key, value in manifest["sources"].items()
        }
        vocab = {
            "sha256": manifest["vocabulary"]["sha256"],
            "size": int(manifest["vocabulary"]["size"]),
            "special_tokens": manifest["vocabulary"]["special_tokens"],
        }
        packing = {
            key: manifest["packing"][key]
            for key in ("schema_version", "mode", "block_size", "transition_policy")
        }
        if reference_sources is None:
            reference_sources = sources
            reference_vocab = vocab
            reference_packing = packing
            reference_homology_policy = homology_policy
        elif sources != reference_sources:
            raise ValueError("genome and genus manifests do not share the same sources")
        elif vocab != reference_vocab:
            raise ValueError("genome and genus manifests do not share the same vocabulary")
        elif packing != reference_packing:
            raise ValueError("genome and genus manifests do not share the same packing policy")
        elif homology_policy != reference_homology_policy:
            raise ValueError("genome and genus manifests do not share the same homology policy")


def build_freeze_index(
    *,
    config_path: Path,
    seed: int,
    manifests: dict[str, tuple[dict[str, Any], Path]],
) -> dict[str, Any]:
    validate_protocol_manifests(manifests)
    manifest_paths = [path.resolve() for _, path in manifests.values()]
    freeze_root = Path(os.path.commonpath([path.parent for path in manifest_paths]))
    protocols = {}
    for protocol in GROUP_PROTOCOLS:
        manifest, manifest_path = manifests[protocol]
        protocols[protocol] = {
            "dataset_id": manifest["dataset"]["id"],
            "manifest_path": os.path.relpath(manifest_path.resolve(), freeze_root),
            "manifest_sha256": file_sha256(manifest_path),
            "record_counts": manifest["split_policy"]["record_counts"],
            "group_counts": manifest["split_policy"]["group_counts"],
            "achieved_record_fractions": manifest["split_policy"][
                "achieved_record_fractions"
            ],
        }
    payload = {
        "schema": {"name": FREEZE_SCHEMA_NAME, "version": FREEZE_SCHEMA_VERSION},
        "config": {
            "path": os.path.relpath(config_path.resolve(), freeze_root),
            "sha256": file_sha256(config_path),
        },
        "split_seed": int(seed),
        "protocols": protocols,
    }
    payload["freeze_id"] = freeze_identity(payload)
    return payload


def freeze_identity(index: dict[str, Any]) -> str:
    """Return a location-independent identity for a genome/genus freeze pair."""
    payload = copy.deepcopy(index)
    payload.pop("freeze_id", None)
    payload.get("config", {}).pop("path", None)
    for protocol in payload.get("protocols", {}).values():
        protocol.pop("manifest_path", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def freeze_contract(index: dict[str, Any], *, release: str) -> dict[str, Any]:
    """Extract the portable, reviewable approval contract from a freeze index."""
    return {
        "schema": {
            "name": CONTRACT_SCHEMA_NAME,
            "version": CONTRACT_SCHEMA_VERSION,
        },
        "release": release,
        "dataset_freeze_schema": index["schema"],
        "dataset_freeze_id": index["freeze_id"],
        "source_config_sha256": index["config"]["sha256"],
        "split_seed": int(index["split_seed"]),
        "protocols": {
            protocol: {
                key: index["protocols"][protocol][key]
                for key in (
                    "dataset_id",
                    "manifest_sha256",
                    "record_counts",
                    "group_counts",
                    "achieved_record_fractions",
                )
            }
            for protocol in GROUP_PROTOCOLS
        },
    }


def validate_freeze_contract(index: dict[str, Any], contract: dict[str, Any]) -> None:
    schema = contract.get("schema", {})
    if schema != {"name": CONTRACT_SCHEMA_NAME, "version": CONTRACT_SCHEMA_VERSION}:
        raise ValueError(f"unsupported pipeline freeze contract schema: {schema!r}")
    expected = freeze_contract(index, release=str(contract.get("release", "")))
    if contract != expected:
        raise ValueError("pipeline freeze contract does not match the dataset freeze")


def load_and_validate_freeze(path: Path) -> dict[str, Any]:
    """Validate a freeze index and every bound manifest/artifact."""
    resolved = path.expanduser().resolve()
    index = json.loads(resolved.read_text())
    expected_schema = {"name": FREEZE_SCHEMA_NAME, "version": FREEZE_SCHEMA_VERSION}
    if index.get("schema") != expected_schema:
        raise ValueError(
            f"unsupported dataset freeze schema: {index.get('schema')!r}; "
            f"expected {expected_schema!r}"
        )
    if index.get("freeze_id") != freeze_identity(index):
        raise ValueError("dataset freeze identity mismatch")

    config_path = (resolved.parent / index["config"]["path"]).resolve()
    if file_sha256(config_path) != index["config"]["sha256"]:
        raise ValueError("dataset freeze source config hash mismatch")

    manifests = {}
    for protocol in GROUP_PROTOCOLS:
        declared = index["protocols"][protocol]
        manifest_path = (resolved.parent / declared["manifest_path"]).resolve()
        if file_sha256(manifest_path) != declared["manifest_sha256"]:
            raise ValueError(f"{protocol} manifest hash mismatch")
        manifest = load_dataset_manifest(manifest_path)
        if manifest["dataset"]["id"] != declared["dataset_id"]:
            raise ValueError(f"{protocol} dataset identity mismatch")
        for key in (
            "record_counts",
            "group_counts",
            "achieved_record_fractions",
        ):
            if manifest["split_policy"][key] != declared[key]:
                raise ValueError(f"{protocol} {key} mismatch")
        manifests[protocol] = (manifest, manifest_path)
    validate_protocol_manifests(manifests)
    return index


def _run_builder(
    *,
    config_path: Path,
    freeze_id: str,
    protocol: str,
    seed: int,
    output_root: Path,
    run_root: Path,
    mmseqs_executable: str,
    nucleotide_executable: str,
    audit_threads: int,
) -> Path:
    output_dir = output_root / freeze_id / protocol
    run_dir = run_root / freeze_id / protocol
    if output_dir.exists() or run_dir.exists():
        raise ValueError(
            f"refusing to overwrite freeze output for {protocol}; choose a new "
            "freeze id or remove the incomplete artifacts explicitly"
        )
    command = [
        sys.executable,
        "-m",
        "scripts.build_global_manifest",
        "--config",
        str(config_path),
        "--run-id",
        f"{freeze_id}-{protocol}",
        "--run-dir",
        str(run_dir),
        "--output-dir",
        str(output_dir),
        "--group-by",
        protocol,
        "--seed",
        str(seed),
        "--mmseqs-executable",
        mmseqs_executable,
        "--audit-threads",
        str(audit_threads),
        "--nucleotide-executable",
        nucleotide_executable,
    ]
    subprocess.run(command, check=True)
    return output_dir / "manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--freeze-id", required=True)
    parser.add_argument("--output-root", default="data/processed/corrected")
    parser.add_argument("--run-root", default="runs/dataset_freeze")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--mmseqs-executable", default="mmseqs")
    parser.add_argument("--nucleotide-executable", default="minimap2")
    parser.add_argument("--audit-threads", type=int, default=1)
    parser.add_argument(
        "--verify-sources-only",
        action="store_true",
        help="Verify the pinned inventory without building derived datasets.",
    )
    parser.add_argument(
        "--finalize-existing",
        action="store_true",
        help="Rebuild only freeze.json from already completed protocol manifests.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = load_and_validate_source_config(config_path)
    print(f"[freeze] Verified {len(cfg['datasets'])} pinned source files.")
    if args.verify_sources_only:
        return
    output_root = Path(args.output_root)
    run_root = Path(args.run_root)
    if args.finalize_existing:
        manifest_paths = {
            protocol: output_root / args.freeze_id / protocol / "manifest.json"
            for protocol in GROUP_PROTOCOLS
        }
        missing = [str(path) for path in manifest_paths.values() if not path.is_file()]
        if missing:
            raise ValueError(f"cannot finalize missing protocol manifests: {missing}")
    else:
        if shutil.which(args.mmseqs_executable) is None:
            raise SystemExit(
                f"[error] MMseqs2 executable not found: {args.mmseqs_executable}; "
                "the scientific freeze cannot skip the homology gate"
            )
        if shutil.which(args.nucleotide_executable) is None:
            raise SystemExit(
                f"[error] nucleotide aligner not found: {args.nucleotide_executable}; "
                "the scientific freeze cannot skip nucleotide nearest-neighbor mapping"
            )
        manifest_paths = {
            protocol: _run_builder(
                config_path=config_path,
                freeze_id=args.freeze_id,
                protocol=protocol,
                seed=args.seed,
                output_root=output_root,
                run_root=run_root,
                mmseqs_executable=args.mmseqs_executable,
                nucleotide_executable=args.nucleotide_executable,
                audit_threads=args.audit_threads,
            )
            for protocol in GROUP_PROTOCOLS
        }
    manifests = {
        protocol: (load_dataset_manifest(path), path)
        for protocol, path in manifest_paths.items()
    }
    index = build_freeze_index(
        config_path=config_path,
        seed=args.seed,
        manifests=manifests,
    )
    index_path = output_root / args.freeze_id / "freeze.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(f"[freeze] Completed {index['freeze_id']}: {index_path}")


if __name__ == "__main__":
    main()
