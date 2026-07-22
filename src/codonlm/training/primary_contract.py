"""Fail-closed contract for corrected primary CodonLM training configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


SCHEMA_NAME = "codonlm_primary_training_config"
SCHEMA_VERSION = 1
RELEASE = "corrected-codonlm-v1"
DATASET_FREEZE_ID = "1582505ae40445422711fa15918ee9c229caf84b1b3feba1a71f078259892249"

DATASETS = {
    "genome": {
        "dataset_id": "da3dfce28b7a46b8640d75c7cb417c867137a99e004ea359d85784ff0c269db9",
        "root": "data/processed/corrected/corrected-codonlm-v1/genome",
    },
    "genus": {
        "dataset_id": "10f41e818182704bbe4f95fbd81eb8696047762a32f84d167a4101675945ab95",
        "root": "data/processed/corrected/corrected-codonlm-v1/genus",
    },
}

COMMON_VALUES: dict[str, Any] = {
    "block_size": 512,
    "vocab_size": 68,
    "n_layer": 10,
    "n_head": 8,
    "n_embd": 384,
    "dropout": 0.1,
    "label_smoothing": 0.05,
    "tie_embeddings": True,
    "use_sdpa": True,
    "sep_mask_enabled": True,
    "n_kv_head": None,
    "use_rope": False,
    "use_swiglu": False,
    "use_shape_guidance": False,
    "unfreeze_encoder": False,
    "multi_offset_loss_enabled": False,
    "multi_offset_targets": [],
    "termination_loss_enabled": False,
    "replay_loss_enabled": False,
    "freeze_backbone": False,
    "eos_loss_weight": 1.0,
    "transfer_from": None,
    "batch_size": 4,
    "grad_accum_steps": 32,
    "lr": 0.0003,
    "lr_embedding": 0.0003,
    "min_lr": 0.00003,
    "weight_decay": 0.05,
    "warmup_steps": 100,
    "optimizer": "adamw",
    "scheduler": "cosine",
    "early_stop_patience": 0,
    "max_nonfinite_accumulation_groups": 0,
    "checkpoint_every_steps": 0,
    "checkpoint_every_minutes": 30,
    "save_epochs": False,
    "device": "mps",
    "force_gpu": True,
    "amp": True,
    "use_checkpoint": True,
    "use_mmap": True,
    "bucket_batching": False,
    "num_workers": 0,
    "pin_memory": False,
    "compile": False,
    "out_dir": "outputs/checkpoints",
    "scores_dir": "outputs/scores",
}

ALLOWED_KEYS = frozenset(
    {
        "primary_training_contract",
        "dataset_manifest",
        "itos_path",
        "train_npz",
        "val_npz",
        "test_npz",
        "run_id",
        "seed",
        "dataloader_seed",
        "epochs",
        "max_time_minutes",
        *COMMON_VALUES,
    }
)


def _require_equal(cfg: Mapping[str, Any], key: str, expected: Any) -> None:
    if key not in cfg:
        raise ValueError(f"primary config is missing required key {key!r}")
    if cfg[key] != expected:
        raise ValueError(
            f"primary config key {key!r} must be {expected!r}, got {cfg[key]!r}"
        )


def validate_primary_training_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a corrected primary or pilot config without requiring local data."""
    contract = cfg.get("primary_training_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("missing primary_training_contract mapping")
    expected_header = {
        "schema": SCHEMA_NAME,
        "version": SCHEMA_VERSION,
        "release": RELEASE,
        "dataset_freeze_id": DATASET_FREEZE_ID,
    }
    for key, expected in expected_header.items():
        if contract.get(key) != expected:
            raise ValueError(
                f"primary_training_contract.{key} must be {expected!r}, "
                f"got {contract.get(key)!r}"
            )

    role = contract.get("role")
    protocol = contract.get("protocol")
    if role not in {"pilot", "primary"}:
        raise ValueError("primary_training_contract.role must be 'pilot' or 'primary'")
    if protocol not in DATASETS:
        raise ValueError("primary_training_contract.protocol must be 'genome' or 'genus'")
    if role == "pilot" and protocol != "genome":
        raise ValueError("the corrected primary pilot must use the genome protocol")

    dataset = DATASETS[str(protocol)]
    if contract.get("dataset_id") != dataset["dataset_id"]:
        raise ValueError("primary training dataset_id does not match the frozen protocol")

    unknown = sorted(set(cfg) - ALLOWED_KEYS)
    if unknown:
        raise ValueError(f"undeclared primary config keys are not allowed: {unknown}")
    for key, expected in COMMON_VALUES.items():
        _require_equal(cfg, key, expected)

    root = dataset["root"]
    paths = {
        "dataset_manifest": f"{root}/manifest.json",
        "itos_path": f"{root}/itos.txt",
        "train_npz": f"{root}/train_bs512.npz",
        "val_npz": f"{root}/val_bs512.npz",
        "test_npz": f"{root}/test_bs512.npz",
    }
    for key, expected in paths.items():
        _require_equal(cfg, key, expected)

    seed = int(cfg.get("seed", -1))
    allowed_seeds = {1337} if protocol == "genus" or role == "pilot" else {1337, 2027}
    if seed not in allowed_seeds:
        raise ValueError(f"unsupported {role} seed {seed} for {protocol} protocol")
    _require_equal(cfg, "dataloader_seed", seed)

    if role == "pilot":
        _require_equal(cfg, "epochs", 1)
        _require_equal(cfg, "max_time_minutes", 30)
        expected_run_id = "corrected-codonlm-v1-pilot-genome-seed1337"
    else:
        _require_equal(cfg, "epochs", 10)
        _require_equal(cfg, "max_time_minutes", None)
        expected_run_id = f"corrected-codonlm-v1-{protocol}-seed{seed}"
    _require_equal(cfg, "run_id", expected_run_id)
    return {
        "role": role,
        "protocol": protocol,
        "seed": seed,
        "run_id": expected_run_id,
        "dataset_id": dataset["dataset_id"],
        "dataset_freeze_id": DATASET_FREEZE_ID,
    }


def load_and_validate_primary_training_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    cfg = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"training config must contain a YAML mapping: {config_path}")
    return validate_primary_training_config(cfg)


__all__ = [
    "DATASET_FREEZE_ID",
    "DATASETS",
    "RELEASE",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "load_and_validate_primary_training_config",
    "validate_primary_training_config",
]
