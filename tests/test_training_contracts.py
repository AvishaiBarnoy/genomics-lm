from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch

from src.training.contracts import (
    TRAINING_CONTRACT_VERSION,
    EngineEvent,
    EngineState,
    MetricValue,
    StepContext,
    StepOutput,
    TrainingCallback,
    TrainingCheckpoint,
    TrainingPhase,
    TrainingTask,
    UpdateResult,
    UpdateStrategy,
)


class SyntheticTask:
    def begin_phase(self, phase, epoch):
        self.phase = phase

    def train_batches(self, epoch):
        return [torch.tensor([float(epoch + 1)])]

    def validation_batches(self, epoch):
        return [torch.tensor([float(epoch + 1)])]

    def training_step(self, batch, context):
        return StepOutput(batch.mean(), {"loss": MetricValue(float(batch.mean()))})

    def validation_step(self, batch, context):
        return self.training_step(batch, context)

    def state_dict(self) -> Mapping[str, object]:
        return {"kind": "synthetic"}

    def load_state_dict(self, state):
        assert state["kind"] == "synthetic"


class MultiOptimizerStrategy:
    """Contract fixture representing NoProp-style multiple optimizer updates."""

    def __init__(self):
        self.group_size = 0

    def begin_group(self, group_size):
        self.group_size = group_size

    def process_microbatch(self, task, batch, context):
        return task.training_step(batch, context)

    def commit_group(self):
        return UpdateResult(committed=True, optimizer_steps=3)

    def abort_group(self, reason):
        return UpdateResult(committed=False, optimizer_steps=0, reason=reason)

    def end_epoch(self, metrics):
        pass

    def state_dict(self):
        return {"group_size": self.group_size, "optimizers": [{}, {}, {}]}

    def load_state_dict(self, state):
        self.group_size = state["group_size"]


class EventRecorder:
    def __init__(self):
        self.events = []

    def on_event(self, event):
        self.events.append(event)


def test_contracts_are_model_agnostic_and_runtime_checkable():
    task = SyntheticTask()
    strategy = MultiOptimizerStrategy()
    callback = EventRecorder()

    assert TRAINING_CONTRACT_VERSION == 1
    assert isinstance(task, TrainingTask)
    assert isinstance(strategy, UpdateStrategy)
    assert isinstance(callback, TrainingCallback)

    context = StepContext(
        phase=TrainingPhase.TRAIN,
        epoch=0,
        microbatch=0,
        optimizer_step=0,
        device=torch.device("cpu"),
        group_size=2,
    )
    strategy.begin_group(context.group_size)
    output = strategy.process_microbatch(task, torch.tensor([2.0]), context)
    output.validate()
    assert strategy.commit_group().optimizer_steps == 3

    callback.on_event(EngineEvent("group_committed", context, output.metrics))
    assert callback.events[0].name == "group_committed"


def test_step_output_rejects_non_scalar_loss_and_negative_units():
    with pytest.raises(ValueError, match="scalar"):
        StepOutput(torch.ones(2)).validate()
    with pytest.raises(ValueError, match="non-negative"):
        StepOutput(torch.tensor(1.0), committed_units={"tokens": -1}).validate()


def test_context_and_update_result_validate_engine_invariants():
    with pytest.raises(ValueError, match="group_size"):
        StepContext(
            TrainingPhase.TRAIN, 0, 0, 0, torch.device("cpu"), group_size=0
        )
    with pytest.raises(ValueError, match="aborted"):
        UpdateResult(committed=False, optimizer_steps=1)


def test_training_checkpoint_round_trip_keeps_state_namespaced():
    checkpoint = TrainingCheckpoint(
        engine=EngineState(1, 2, 3, 4),
        task={"model": {"weight": torch.tensor([1.0])}},
        strategy={"optimizers": [{"step": 4}]},
        rng={"python": "state"},
        metadata={"run_fingerprint": "abc"},
    )

    restored = TrainingCheckpoint.from_payload(checkpoint.to_payload())

    assert restored.engine == checkpoint.engine
    assert torch.equal(restored.task["model"]["weight"], torch.tensor([1.0]))
    assert restored.strategy == checkpoint.strategy
    assert restored.metadata["run_fingerprint"] == "abc"


def test_training_checkpoint_rejects_legacy_or_unknown_layouts():
    with pytest.raises(ValueError, match="contract version"):
        TrainingCheckpoint.from_payload({"model_state_dict": {}})
    with pytest.raises(ValueError, match="contract version"):
        TrainingCheckpoint.from_payload({"training_contract_version": 999})
