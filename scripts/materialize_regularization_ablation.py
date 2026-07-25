#!/usr/bin/env python3
"""Materialize matched diagnostic configs without mutating primary contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import yaml

from src.codonlm.training.primary_contract import validate_primary_training_config


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def materialize(matrix_path: Path, output_dir: Path) -> dict:
    matrix_path = matrix_path.expanduser().resolve()
    matrix = yaml.safe_load(matrix_path.read_text())
    if int(matrix.get("schema_version", 0)) != 1:
        raise ValueError("unsupported regularization-ablation schema")
    base_path = Path(matrix["base_config"])
    if not base_path.is_absolute():
        base_path = Path.cwd() / base_path
    base_path = base_path.resolve()
    base = yaml.safe_load(base_path.read_text())
    validate_primary_training_config(base)

    allowed = frozenset(matrix["allowed_overrides"])
    expected_allowed = frozenset({"label_smoothing", "dropout", "tie_embeddings"})
    if allowed != expected_allowed:
        raise ValueError(
            f"allowed_overrides must be exactly {sorted(expected_allowed)}"
        )
    epochs = int(matrix["epochs"])
    total_steps = int(matrix["scheduler_total_steps"])
    if epochs < 1 or total_steps != epochs * 500:
        raise ValueError("diagnostic scheduler horizon must equal epochs * 500")

    output_dir.mkdir(parents=True, exist_ok=True)
    resolved = []
    names = set()
    for variant in matrix["variants"]:
        name = str(variant["name"])
        if not re.fullmatch(r"[a-z0-9_]+", name) or name in names:
            raise ValueError(f"invalid or duplicate variant name: {name}")
        names.add(name)
        overrides = dict(variant["overrides"])
        unexpected = set(overrides) - allowed
        if unexpected:
            raise ValueError(f"variant {name} has undeclared overrides: {sorted(unexpected)}")

        config = dict(base)
        primary_contract = config.pop("primary_training_contract")
        config["diagnostic_experiment_contract"] = {
            "schema_version": 1,
            "experiment": matrix["experiment"],
            "variant": name,
            "base_config": str(base_path),
            "base_config_sha256": _sha256(base_path),
            "base_primary_contract": primary_contract,
            "matrix": str(matrix_path),
            "matrix_sha256": _sha256(matrix_path),
            "expected_nonpad_tokens": int(matrix["expected_nonpad_tokens"]),
            "allowed_overrides": sorted(allowed),
        }
        config.update(overrides)
        config["run_id"] = f"{matrix['experiment']}-{name}"
        config["seed"] = int(matrix["seed"])
        config["dataloader_seed"] = int(matrix["seed"])
        config["epochs"] = epochs
        config["scheduler_total_steps"] = total_steps
        config["max_time_minutes"] = None
        config["early_stop_patience"] = 0
        config["save_epochs"] = False
        config["transfer_from"] = None

        output = output_dir / f"{name}.yaml"
        output.write_text(yaml.safe_dump(config, sort_keys=False))
        resolved.append(
            {
                "name": name,
                "config": str(output.resolve()),
                "sha256": _sha256(output),
                "run_id": config["run_id"],
                "overrides": overrides,
            }
        )

    report = {
        "schema_version": 1,
        "experiment": matrix["experiment"],
        "matrix": str(matrix_path),
        "matrix_sha256": _sha256(matrix_path),
        "base_config": str(base_path),
        "base_config_sha256": _sha256(base_path),
        "epochs": epochs,
        "scheduler_total_steps": total_steps,
        "expected_nonpad_tokens": int(matrix["expected_nonpad_tokens"]),
        "variants": resolved,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path("configs/corrected_regularization_ablation.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = materialize(args.matrix, args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
