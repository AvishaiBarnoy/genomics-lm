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
