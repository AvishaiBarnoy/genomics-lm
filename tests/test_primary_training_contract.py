from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from src.codonlm.training.primary_contract import (
    load_and_validate_primary_training_config,
    validate_primary_training_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = (
    "corrected_primary_pilot_genome_seed1337_v1.yaml",
    "corrected_primary_genome_seed1337_v1.yaml",
    "corrected_primary_genome_seed2027_v1.yaml",
    "corrected_primary_genus_seed1337_v1.yaml",
)


@pytest.mark.parametrize("name", CONFIGS)
def test_tracked_primary_configs_pass_contract(name):
    result = load_and_validate_primary_training_config(ROOT / "configs" / name)
    assert result["run_id"].startswith("corrected-codonlm-v1-")


def _genome_config():
    path = ROOT / "configs" / "corrected_primary_genome_seed1337_v1.yaml"
    return yaml.safe_load(path.read_text())


@pytest.mark.parametrize(
    ("key", "value"),
    (
        ("transfer_from", "legacy.pt"),
        ("n_kv_head", 4),
        ("use_rope", True),
        ("use_swiglu", True),
        ("use_shape_guidance", True),
        ("multi_offset_loss_enabled", True),
        ("termination_loss_enabled", True),
        ("replay_loss_enabled", True),
        ("early_stop_patience", 2),
        ("epochs", 9),
    ),
)
def test_contract_rejects_primary_drift(key, value):
    cfg = deepcopy(_genome_config())
    cfg[key] = value
    with pytest.raises(ValueError, match=key):
        validate_primary_training_config(cfg)


def test_contract_rejects_undeclared_objective_or_architecture_key():
    cfg = deepcopy(_genome_config())
    cfg["protein_critic_loss_enabled"] = True
    with pytest.raises(ValueError, match="undeclared primary config keys"):
        validate_primary_training_config(cfg)


def test_genome_replicates_differ_only_in_seed_identity():
    first = yaml.safe_load(
        (ROOT / "configs" / "corrected_primary_genome_seed1337_v1.yaml").read_text()
    )
    second = yaml.safe_load(
        (ROOT / "configs" / "corrected_primary_genome_seed2027_v1.yaml").read_text()
    )
    for cfg in (first, second):
        cfg.pop("seed")
        cfg.pop("dataloader_seed")
        cfg.pop("run_id")
    assert first == second
