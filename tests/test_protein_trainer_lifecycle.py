import json
from pathlib import Path

import torch
import yaml

from src.protein_lm.train_lm import train
from src.protein_lm.train_classifier import train_classifier


def _write_config(tmp_path: Path, epochs: int) -> Path:
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "validation.jsonl"
    records = [{"sequence": "MKTAA"}, {"sequence": "MQQVV"}]
    text = "".join(json.dumps(record) + "\n" for record in records)
    train_path.write_text(text)
    val_path.write_text(text)
    config = {
        "run_id": "protein-lm-smoke",
        "model": {
            "n_layer": 1,
            "n_head": 1,
            "n_embd": 16,
            "block_size": 8,
            "dropout": 0.0,
        },
        "training": {
            "batch_size": 2,
            "lr": 1e-3,
            "epochs": epochs,
            "grad_accum_steps": 1,
            "num_workers": 0,
            "seed": 7,
        },
        "data": {"train_path": str(train_path), "val_path": str(val_path)},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_protein_lm_serial_launch_and_completed_extension(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = _write_config(tmp_path, epochs=1)
    train(str(config))
    first = tmp_path / "runs" / "protein_lm" / "protein-lm-smoke"
    checkpoint = first / "checkpoints" / "last.pt"
    assert (first / "run_complete.json").exists()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["run_progress"]["completed_epochs"] == 1
    assert payload["training_contract_version"] == 1
    assert payload["model_state_dict"].keys() == payload["task"]["model"].keys()
    for name, tensor in payload["model_state_dict"].items():
        assert torch.equal(tensor, payload["task"]["model"][name])
    assert payload["optimizer_state_dict"].keys() == payload["strategy"]["optimizer"].keys()
    assert payload["scheduler_state_dict"] == payload["strategy"]["scheduler"]

    train(str(config))
    assert (tmp_path / "runs" / "protein_lm" / "protein-lm-smoke-r002").exists()

    config = _write_config(tmp_path, epochs=2)
    train(str(config), resume=str(checkpoint), run_id="protein-lm-smoke")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["run_progress"]["completed_epochs"] == 2
    assert (first / "run_complete_epoch_001.json").exists()


def test_protein_classifier_uses_engine_checkpoint_and_label_contract(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    train_path = tmp_path / "classifier_train.jsonl"
    val_path = tmp_path / "classifier_validation.jsonl"
    records = [
        {"sequence": "MKTAA", "func_label": "enzyme"},
        {"sequence": "MQQVV", "func_label": "other"},
        {"sequence": "MKTAV", "func_label": "enzyme"},
        {"sequence": "MQQVA", "func_label": "other"},
    ]
    text = "".join(json.dumps(record) + "\n" for record in records)
    train_path.write_text(text)
    val_path.write_text(text)
    config = {
        "run_id": "classifier-smoke",
        "model": {
            "n_layer": 1,
            "n_head": 1,
            "n_embd": 8,
            "block_size": 8,
            "dropout": 0.0,
            "num_classes": 2,
        },
        "training": {
            "batch_size": 2,
            "lr": 1e-3,
            "epochs": 1,
            "grad_accum_steps": 1,
            "num_workers": 0,
            "seed": 7,
            "log_every_steps": 0,
        },
        "data": {"train_path": str(train_path), "val_path": str(val_path)},
    }
    config_path = tmp_path / "classifier.yaml"
    config_path.write_text(yaml.safe_dump(config))

    result = train_classifier(str(config_path))

    run = tmp_path / "runs" / "protein_classifier" / "classifier-smoke"
    payload = torch.load(
        run / "checkpoints" / "last.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert result.status == "complete"
    assert payload["training_contract_version"] == 1
    assert payload["run_progress"]["completed_epochs"] == 1
    assert payload["label_map"] == {"enzyme": 0, "other": 1}
    assert payload["task"]["label_map"] == payload["label_map"]
    assert payload["model_state_dict"].keys() == payload["task"]["model"].keys()
    assert payload["scheduler_state_dict"] == payload["strategy"]["scheduler"]
    assert payload["loss"] == payload["metadata"]["metrics"]["loss"]
    assert (run / "checkpoints" / "epoch_001.pt").exists()
    assert (run / "run_complete.json").exists()
