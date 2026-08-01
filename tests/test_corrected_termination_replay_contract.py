from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


def _config():
    return yaml.safe_load(
        (ROOT / "configs" / "corrected_termination_replay_seed1337.yaml").read_text()
    )


def test_corrected_replay_config_preserves_locked_protocol():
    cfg = _config()

    assert cfg["transfer_from"].endswith(
        "corrected-termination-head-seed1337/checkpoints/best.pt"
    )
    assert cfg["replay_loss_enabled"] is True
    assert cfg["replay_every_microbatches"] == cfg["grad_accum_steps"] == 16
    assert cfg["replay_loss_weight"] / cfg["grad_accum_steps"] == pytest.approx(0.2)
    assert cfg["extension_experiment_contract"]["replay_source_split"] == "train"
    assert cfg["extension_experiment_contract"]["replay_failure_records"] == 79
    assert len(cfg["replay_class_weights"]) == cfg["termination_n_classes"]


def test_corrected_replay_config_keeps_other_extensions_disabled():
    cfg = _config()

    assert cfg["multi_offset_loss_enabled"] is False
    assert cfg["use_shape_guidance"] is False
    assert cfg["freeze_backbone"] is False
