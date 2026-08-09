from __future__ import annotations

import copy
import json

import numpy as np
import torch

from src.protein_lm.mlp_heads_task import HeadBatch, MLPHeadsTask
from src.protein_lm.train_mlp_heads import (
    MultiTaskMLPClassifier,
    train_mlp_heads,
)
from src.training.contracts import StepContext, TrainingPhase


def _task(model, batches, task_dims):
    loaders = {
        name: [
            {"X": batch.features, "label": batch.labels}
            for batch in batches
            if batch.task_name == name
        ]
        for name in task_dims
    }
    return MLPHeadsTask(
        model=model,
        train_loaders=loaders,
        validation_loaders=loaders,
        device=torch.device("cpu"),
        train_generators={name: torch.Generator() for name in task_dims},
        seed=11,
        task_dims=task_dims,
    )


def test_combined_optimizer_matches_independent_head_optimizers():
    task_dims = {"family": 2, "function": 3}
    batches = [
        HeadBatch("family", torch.randn(3, 4), torch.tensor([0, 1, 0])),
        HeadBatch("function", torch.randn(3, 4), torch.tensor([0, 1, 2])),
    ]
    torch.manual_seed(13)
    combined_model = MultiTaskMLPClassifier(4, task_dims)
    independent_model = copy.deepcopy(combined_model)
    combined_task = _task(combined_model, batches, task_dims)
    independent_task = _task(independent_model, batches, task_dims)
    combined_optimizer = torch.optim.AdamW(
        combined_model.parameters(), lr=1e-3, weight_decay=0.01
    )
    independent_optimizers = {
        name: torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
        for name, head in independent_model.heads.items()
    }
    context = StepContext(TrainingPhase.TRAIN, 0, 0, 0, torch.device("cpu"))

    combined_task.begin_phase(TrainingPhase.TRAIN, 0)
    independent_task.begin_phase(TrainingPhase.TRAIN, 0)
    for batch in batches:
        torch.manual_seed(29)
        combined_optimizer.zero_grad(set_to_none=True)
        combined_task.training_step(batch, context).loss.backward()
        combined_optimizer.step()

        torch.manual_seed(29)
        optimizer = independent_optimizers[batch.task_name]
        optimizer.zero_grad(set_to_none=True)
        independent_task.training_step(batch, context).loss.backward()
        optimizer.step()

    for actual, expected in zip(
        combined_model.parameters(), independent_model.parameters(), strict=True
    ):
        assert torch.allclose(actual, expected)


def _write_fixture(tmp_path):
    rng = np.random.default_rng(7)
    train = tmp_path / "train.npz"
    validation = tmp_path / "validation.npz"
    arrays = {
        "X": rng.normal(size=(12, 4)).astype(np.float32),
        "y_family": np.array([0, 1] * 6),
        "y_function": np.array([0, 1, 2] * 4),
        "y_stability": np.array([1, 0] * 6),
    }
    np.savez(train, **arrays)
    np.savez(validation, **arrays)
    vocabs = tmp_path / "vocabs.json"
    vocabs.write_text(
        json.dumps(
            {
                "pfam": {"a": 0, "b": 1},
                "ec": {"a": 0, "b": 1, "c": 2},
                "stability": {"stable": 0, "unstable": 1},
            }
        )
    )
    return train, validation, vocabs


def test_trainer_creates_collision_safe_selected_head_artifact(tmp_path):
    train, validation, vocabs = _write_fixture(tmp_path)
    root = tmp_path / "runs"
    first = train_mlp_heads(
        train,
        validation,
        vocabs,
        epochs=1,
        batch_size=4,
        out_dir=root,
        device_name="cpu",
    )
    second = train_mlp_heads(
        train,
        validation,
        vocabs,
        epochs=1,
        batch_size=4,
        out_dir=root,
        device_name="cpu",
    )

    assert first.status == second.status == "complete"
    assert (root / "mlp-heads" / "checkpoints" / "mlp_heads.pt").is_file()
    assert (root / "mlp-heads-r002" / "checkpoints" / "mlp_heads.pt").is_file()
    checkpoint = torch.load(
        root / "mlp-heads" / "checkpoints" / "last.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert set(checkpoint["task"]["best_head_states"]) == {
        "family",
        "function",
        "stability",
    }


def test_dataset_rejects_out_of_vocabulary_labels(tmp_path):
    train, _, _ = _write_fixture(tmp_path)
    with np.load(train) as data:
        arrays = {name: data[name] for name in data.files}
    arrays["y_family"] = np.full(12, 2)
    invalid = tmp_path / "invalid.npz"
    np.savez(invalid, **arrays)

    from src.protein_lm.train_mlp_heads import TaskFeatureDataset

    try:
        TaskFeatureDataset(invalid, "family", num_classes=2)
    except ValueError as exc:
        assert "exceed its vocabulary" in str(exc)
    else:
        raise AssertionError("out-of-range labels were accepted")


def test_interrupted_resume_matches_uninterrupted_training(tmp_path):
    train, validation, vocabs = _write_fixture(tmp_path)
    reference_root = tmp_path / "reference"
    resumed_root = tmp_path / "resumed"
    train_mlp_heads(
        train,
        validation,
        vocabs,
        epochs=1,
        batch_size=4,
        out_dir=reference_root,
        run_id="reference",
        device_name="cpu",
    )
    interrupted = train_mlp_heads(
        train,
        validation,
        vocabs,
        epochs=1,
        batch_size=4,
        out_dir=resumed_root,
        run_id="interrupted",
        device_name="cpu",
        max_time_minutes=0,
    )
    assert interrupted.status == "interrupted"
    last = resumed_root / "interrupted" / "checkpoints" / "last.pt"
    resumed = train_mlp_heads(
        train,
        validation,
        vocabs,
        epochs=1,
        batch_size=4,
        out_dir=resumed_root,
        run_id="interrupted",
        resume=last,
        device_name="cpu",
    )
    assert resumed.status == "complete"

    reference = torch.load(
        reference_root / "reference" / "checkpoints" / "last.pt",
        map_location="cpu",
        weights_only=False,
    )
    actual = torch.load(last, map_location="cpu", weights_only=False)
    for name, tensor in reference["task"]["model"].items():
        assert torch.equal(actual["task"]["model"][name], tensor)
    for parameter_id, state in reference["strategy"]["optimizer"]["state"].items():
        actual_state = actual["strategy"]["optimizer"]["state"][parameter_id]
        for state_name, value in state.items():
            if torch.is_tensor(value):
                assert torch.equal(actual_state[state_name], value)
            else:
                assert actual_state[state_name] == value
