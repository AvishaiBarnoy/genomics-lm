from pathlib import Path

import pytest
import yaml

from scripts.materialize_regularization_ablation import materialize


def test_regularization_ablation_materializes_only_declared_differences(tmp_path):
    report = materialize(
        Path("configs/corrected_regularization_ablation.yaml"),
        tmp_path,
    )
    assert len(report["variants"]) == 4
    assert report["expected_nonpad_tokens"] == 50_476_876

    configs = [
        yaml.safe_load(Path(item["config"]).read_text())
        for item in report["variants"]
    ]
    for config in configs:
        assert "primary_training_contract" not in config
        assert config["diagnostic_experiment_contract"]["base_primary_contract"]
        assert config["epochs"] == 2
        assert config["scheduler_total_steps"] == 1000
        assert config["transfer_from"] is None

    ignored = {
        "diagnostic_experiment_contract",
        "run_id",
        "label_smoothing",
        "dropout",
        "tie_embeddings",
    }
    reference = {key: value for key, value in configs[0].items() if key not in ignored}
    for config in configs[1:]:
        comparable = {key: value for key, value in config.items() if key not in ignored}
        assert comparable == reference


def test_regularization_ablation_rejects_undeclared_override(tmp_path):
    source = yaml.safe_load(
        Path("configs/corrected_regularization_ablation.yaml").read_text()
    )
    source["variants"][0]["overrides"]["n_layer"] = 11
    matrix = tmp_path / "invalid.yaml"
    matrix.write_text(yaml.safe_dump(source))
    with pytest.raises(ValueError, match="undeclared overrides"):
        materialize(matrix, tmp_path / "out")
