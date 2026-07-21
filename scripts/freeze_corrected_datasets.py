#!/usr/bin/env python3
"""Build and bind corrected genome/genus holdout datasets into one freeze index."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from scripts.build_global_manifest import validate_pinned_source
from src.codonlm.dataset_manifest import file_sha256, load_dataset_manifest


FREEZE_SCHEMA_NAME = "codonlm_dataset_freeze"
FREEZE_SCHEMA_VERSION = 1
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
    protocols = {}
    for protocol in GROUP_PROTOCOLS:
        manifest, manifest_path = manifests[protocol]
        protocols[protocol] = {
            "dataset_id": manifest["dataset"]["id"],
            "manifest_path": str(manifest_path.resolve()),
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
            "path": str(config_path.resolve()),
            "sha256": file_sha256(config_path),
        },
        "split_seed": int(seed),
        "protocols": protocols,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["freeze_id"] = hashlib.sha256(encoded).hexdigest()
    return payload


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
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = load_and_validate_source_config(config_path)
    print(f"[freeze] Verified {len(cfg['datasets'])} pinned source files.")
    if args.verify_sources_only:
        return
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

    output_root = Path(args.output_root)
    run_root = Path(args.run_root)
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
