from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from src.codonlm.model_tiny_gpt import TinyGPT
from src.codonlm.training.loop import NonfiniteGroupLimitError, run_training


def _config(tmp_path, *, max_nonfinite_groups: int) -> dict:
    return {
        "vocab_size": 69,
        "block_size": 4,
        "n_layer": 1,
        "n_head": 1,
        "n_embd": 8,
        "dropout": 0.0,
        "batch_size": 1,
        "grad_accum_steps": 2,
        "max_nonfinite_accumulation_groups": max_nonfinite_groups,
        "lr": 0.001,
        "min_lr": 0.0001,
        "weight_decay": 0.0,
        "warmup_steps": 0,
        "epochs": 1,
        "optimizer": "adamw",
        "amp": False,
        "use_checkpoint": False,
        "scheduler": "cosine",
        "early_stop_patience": 2,
        "out_dir": str(tmp_path / "unused-checkpoints"),
        "scores_dir": str(tmp_path / "unused-scores"),
        "seed": 42,
        "num_workers": 0,
    }


def _write_inputs(tmp_path, config: dict):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    x_train = np.ones((5, 4), dtype=np.int32)
    y_train = np.full((5, 4), 2, dtype=np.int32)
    x_val = np.ones((2, 4), dtype=np.int32)
    y_val = np.full((2, 4), 2, dtype=np.int32)
    paths = {}
    for name, x, y in (
        ("train", x_train, y_train),
        ("val", x_val, y_val),
        ("test", x_val, y_val),
    ):
        path = tmp_path / f"{name}.npz"
        np.savez_compressed(path, X=x, Y=y)
        paths[name] = path
    return config_path, paths


def _args(config_path, paths, *, run_id: str, resume=None):
    return SimpleNamespace(
        config=str(config_path),
        run_id=run_id,
        resume=resume,
        transfer_from=None,
        train_npz=[str(paths["train"])],
        val_npz=[str(paths["val"])],
        test_npz=[str(paths["test"])],
    )


def test_nonfinite_abort_checkpoint_and_resume_preserve_step_counters(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    config = _config(tmp_path, max_nonfinite_groups=0)
    config_path, paths = _write_inputs(tmp_path, config)
    original_forward = TinyGPT.forward
    train_calls = 0

    def fail_second_train_microbatch(self, *args, **kwargs):
        nonlocal train_calls
        result = original_forward(self, *args, **kwargs)
        if self.training:
            train_calls += 1
            if train_calls == 2:
                logits, loss = result
                return logits, loss * torch.tensor(float("nan"), device=loss.device)
        return result

    monkeypatch.setattr(TinyGPT, "forward", fail_second_train_microbatch)
    args = _args(config_path, paths, run_id="nonfinite-resume")

    with pytest.raises(NonfiniteGroupLimitError):
        run_training(dict(config), args)

    checkpoint_path = tmp_path / "runs/nonfinite-resume/checkpoints/last.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["step"] == 0
    assert checkpoint["scheduler"]["last_epoch"] == 0
    assert checkpoint["epoch_microbatch_idx"] == 2
    assert checkpoint["accumulation_health"] == {
        "active_microbatches": 0,
        "nonfinite_microbatches": 1,
        "aborted_groups": 1,
        "discarded_finite_microbatches": 1,
    }

    monkeypatch.setattr(TinyGPT, "forward", original_forward)
    resume_args = _args(
        config_path,
        paths,
        run_id="nonfinite-resume",
        resume=str(checkpoint_path),
    )
    run_training(dict(config), resume_args)

    resumed = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert resumed["step"] == 2
    assert resumed["scheduler"]["last_epoch"] == 2
    assert resumed["epoch_microbatch_idx"] == 0
    assert resumed["accumulation_health"]["aborted_groups"] == 1
    assert resumed["accumulation_health"]["discarded_finite_microbatches"] == 1
