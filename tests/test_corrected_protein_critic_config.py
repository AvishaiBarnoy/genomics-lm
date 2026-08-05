from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_corrected_critic_config_freezes_architecture_and_tasks():
    cfg = yaml.safe_load(
        (ROOT / "configs" / "corrected_protein_critic_v1.yaml").read_text()
    )
    assert (cfg["n_layer"], cfg["n_head"], cfg["n_embd"]) == (8, 8, 256)
    assert cfg["bidirectional"] is True
    assert cfg["pooling"] == "attention"
    assert cfg["transfer_from"] is None
    assert cfg["saliency_regularizer_weight"] == 0.0
    assert cfg["regression_tasks"] == ["stability"]
    assert cfg["dataset_manifest"].endswith("corrected-v2/manifest.json")
    assert cfg["batch_size"] == 2
    assert cfg["grad_accum_steps"] == 16
    assert cfg["log_every_steps"] == 200


def test_class_balanced_critic_config_changes_only_training_objective():
    baseline = yaml.safe_load(
        (ROOT / "configs" / "corrected_protein_critic_v1.yaml").read_text()
    )
    ablation = yaml.safe_load(
        (
            ROOT
            / "configs"
            / "corrected_protein_critic_class_balanced_v1.yaml"
        ).read_text()
    )
    allowed_changes = {
        "critic_training_contract",
        "run_id",
        "classification_class_weighting",
        "classification_class_weight_max",
    }
    for key in set(baseline) | set(ablation):
        if key not in allowed_changes:
            assert ablation[key] == baseline[key]
    assert ablation["classification_class_weighting"] == "sqrt_inverse_frequency"
    assert ablation["classification_class_weight_max"] == 4.0
