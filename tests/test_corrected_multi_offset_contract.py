from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _config():
    return yaml.safe_load(
        (ROOT / "configs" / "corrected_multi_offset_heads_seed1337.yaml").read_text()
    )


def test_corrected_multi_offset_probe_preserves_anchor_and_split():
    cfg = _config()
    contract = cfg["extension_experiment_contract"]

    assert cfg["transfer_from"].endswith(
        "corrected-codonlm-v1-batch64-lr-ablation-lr_1_5e4/checkpoints/best.pt"
    )
    assert "corrected/corrected-codonlm-v1/genome" in cfg["dataset_manifest"]
    assert contract["anchor_run_id"] == (
        "corrected-codonlm-v1-batch64-lr-ablation-lr_1_5e4"
    )
    assert contract["dataset_protocol"] == "frozen_genome_holdout"


def test_corrected_multi_offset_probe_is_head_only():
    cfg = _config()
    contract = cfg["extension_experiment_contract"]

    assert cfg["multi_offset_loss_enabled"] is True
    assert cfg["multi_offset_targets"] == [2, 4, 8, 16, 32]
    assert cfg["freeze_backbone"] is True
    assert cfg["termination_loss_enabled"] is False
    assert cfg["replay_loss_enabled"] is False
    assert cfg["use_shape_guidance"] is False
    assert contract["backbone_policy"] == "frozen"
    assert contract["primary_next_token_head_policy"] == "frozen"
    assert contract["training_prior_merge"] is False


def test_corrected_multi_offset_probe_does_not_confound_distance_with_weight():
    cfg = _config()
    weights = cfg["multi_offset_weights"]

    assert set(weights) == set(cfg["multi_offset_targets"])
    assert len(set(weights.values())) == 1
    assert cfg["batch_size"] * cfg["grad_accum_steps"] == 64
    assert cfg["warmup_fraction"] == 0.1
    assert cfg["use_checkpoint"] is False
