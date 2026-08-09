from __future__ import annotations

import pytest
import torch

from src.training.contracts import MetricValue, StepContext, StepOutput, TrainingPhase
from src.training.engine import EngineConfig, TrainingEngine
from src.training.run_lifecycle import TrainingRun
from src.training.strategies import AccumulatedBackpropStrategy, PrecisionPolicy


class LinearTask:
    def __init__(self, values, *, nonfinite_value=None):
        self.model = torch.nn.Linear(1, 1, bias=False)
        self.values = [float(value) for value in values]
        self.nonfinite_value = nonfinite_value
        self.phase = None

    def begin_phase(self, phase, epoch):
        self.phase = phase
        self.model.train(phase == TrainingPhase.TRAIN)

    def train_batches(self, epoch):
        return [torch.tensor([[value]]) for value in self.values]

    def validation_batches(self, epoch):
        return [torch.tensor([[1.0]])]

    def training_step(self, batch, context):
        prediction = self.model(batch)
        loss = prediction.square().mean()
        if self.nonfinite_value is not None and batch.item() == self.nonfinite_value:
            loss = loss * torch.tensor(float("nan"))
        return StepOutput(
            loss,
            {"loss": MetricValue(float(loss.detach()), 1.0)},
            {"examples": len(batch)},
        )

    def validation_step(self, batch, context):
        loss = self.model(batch).square().mean()
        return StepOutput(loss, {"loss": MetricValue(float(loss), 1.0)})

    def state_dict(self):
        return {"model": self.model.state_dict()}

    def load_state_dict(self, state):
        self.model.load_state_dict(state["model"])


class ExpireAfterFirstGroup:
    def __init__(self):
        self.calls = 0

    def expired(self):
        self.calls += 1
        return self.calls == 1


class EventRecorder:
    def __init__(self):
        self.events = []

    def on_event(self, event):
        self.events.append(event)


class WeightedValidationTask(LinearTask):
    def validation_batches(self, epoch):
        return [torch.tensor([[2.0]]), torch.tensor([[4.0]])]

    def validation_step(self, batch, context):
        weight = 100.0 if batch.item() == 2.0 else 10.0
        value = float(batch.item())
        return StepOutput(
            self.model(batch).square().mean(),
            {"score": MetricValue(value * weight, weight)},
        )


def _build_engine(tmp_path, run, task, *, epochs=1, timer=None):
    optimizer = torch.optim.SGD(task.model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: 1.0
    )
    strategy = AccumulatedBackpropStrategy(
        optimizer, scheduler=scheduler, parameters=task.model.parameters()
    )
    engine = TrainingEngine(
        task=task,
        strategy=strategy,
        run=run,
        config=EngineConfig(epochs=epochs, grad_accum_steps=2),
        device=torch.device("cpu"),
        wall_timer=timer,
    )
    return engine, optimizer, scheduler


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("epochs", 1.5),
        ("grad_accum_steps", 2.0),
        ("validate_every_epochs", True),
    ],
)
def test_engine_config_rejects_non_integer_counts(field, value):
    values = {
        "epochs": 1,
        "grad_accum_steps": 1,
        "validate_every_epochs": 1,
    }
    values[field] = value

    with pytest.raises(TypeError, match=f"{field} must be an integer"):
        EngineConfig(**values)


def test_remainder_group_uses_actual_size_and_scheduler_steps_per_update(tmp_path):
    task = LinearTask([1.0, 2.0, 3.0])
    task.model.weight.data.fill_(1.0)
    run = TrainingRun.open(tmp_path, "remainder")
    engine, optimizer, scheduler = _build_engine(tmp_path, run, task)

    result = engine.fit()

    reference = torch.nn.Linear(1, 1, bias=False)
    reference.weight.data.fill_(1.0)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    for values in ([1.0, 2.0], [3.0]):
        reference_optimizer.zero_grad()
        losses = [reference(torch.tensor([[value]])).square().mean() for value in values]
        torch.stack(losses).mean().backward()
        reference_optimizer.step()

    assert result.state.optimizer_step == 2
    assert scheduler.last_epoch == 2
    assert optimizer.param_groups[0]["lr"] == 0.1
    assert torch.allclose(task.model.weight, reference.weight)
    assert (run.checkpoints / "last.pt").exists()
    assert (run.checkpoints / "best.pt").exists()
    run.close()


def test_nonfinite_microbatch_aborts_the_entire_group(tmp_path):
    task = LinearTask([1.0, 2.0, 3.0], nonfinite_value=2.0)
    task.model.weight.data.fill_(1.0)
    run = TrainingRun.open(tmp_path, "nonfinite")
    engine, _, _ = _build_engine(tmp_path, run, task)

    result = engine.fit()

    assert result.aborted_groups == 1
    assert result.state.optimizer_step == 1
    assert task.model.weight.item() == torch.tensor(-0.8).item()
    run.close()


def test_interrupted_resume_matches_uninterrupted_parameters(tmp_path):
    reference_task = LinearTask([1.0, 2.0, 3.0, 4.0])
    reference_task.model.weight.data.fill_(1.0)
    reference_run = TrainingRun.open(tmp_path, "reference")
    reference_engine, _, _ = _build_engine(tmp_path, reference_run, reference_task)
    reference_result = reference_engine.fit()
    reference_weight = reference_task.model.weight.detach().clone()
    reference_run.close()

    interrupted_task = LinearTask([1.0, 2.0, 3.0, 4.0])
    interrupted_task.model.weight.data.fill_(1.0)
    interrupted_run = TrainingRun.open(tmp_path, "resumable")
    interrupted_engine, _, _ = _build_engine(
        tmp_path, interrupted_run, interrupted_task, timer=ExpireAfterFirstGroup()
    )
    interrupted_result = interrupted_engine.fit()
    checkpoint = interrupted_run.checkpoints / "last.pt"
    interrupted_run.close()

    assert interrupted_result.status == "interrupted"
    assert interrupted_result.state.microbatch == 2

    resumed_run = TrainingRun.open(
        tmp_path, "resumable", resume=checkpoint, target_epochs=1
    )
    resumed_task = LinearTask([1.0, 2.0, 3.0, 4.0])
    resumed_engine, _, resumed_scheduler = _build_engine(
        tmp_path, resumed_run, resumed_task
    )
    resumed_result = resumed_engine.fit()

    assert reference_result.state.optimizer_step == resumed_result.state.optimizer_step == 2
    assert resumed_scheduler.last_epoch == 2
    assert torch.allclose(resumed_task.model.weight, reference_weight)
    resumed_run.close()


def test_precision_policy_applies_autocast_without_task_changes():
    task = LinearTask([1.0])
    optimizer = torch.optim.SGD(task.model.parameters(), lr=0.1)
    strategy = AccumulatedBackpropStrategy(
        optimizer,
        precision=PrecisionPolicy(
            device_type="cpu", dtype=torch.bfloat16, enabled=True
        ),
    )
    context = StepContext(
        TrainingPhase.TRAIN, 0, 0, 0, torch.device("cpu")
    )

    strategy.begin_group(1)
    output = strategy.process_microbatch(task, torch.tensor([[1.0]]), context)
    result = strategy.commit_group()

    assert output.loss.dtype == torch.bfloat16
    assert result.committed


def test_validation_metrics_are_weighted_and_emitted_to_callbacks(tmp_path):
    task = WeightedValidationTask([1.0])
    run = TrainingRun.open(tmp_path, "weighted-metrics")
    optimizer = torch.optim.SGD(task.model.parameters(), lr=0.0)
    recorder = EventRecorder()
    engine = TrainingEngine(
        task=task,
        strategy=AccumulatedBackpropStrategy(optimizer),
        run=run,
        config=EngineConfig(epochs=1, monitor="score"),
        device=torch.device("cpu"),
        callbacks=[recorder],
    )

    result = engine.fit()

    expected = (2.0 * 100.0 + 4.0 * 10.0) / 110.0
    validation = next(
        event for event in recorder.events if event.name == "validation_completed"
    )
    assert validation.metrics["score"].total == expected
    assert result.best_metric == expected
    run.close()


def test_strategy_checkpoint_rejects_different_optimizer_class():
    model = torch.nn.Linear(1, 1)
    adamw = AccumulatedBackpropStrategy(
        torch.optim.AdamW(model.parameters(), lr=1e-3)
    )
    state = adamw.state_dict()
    sgd = AccumulatedBackpropStrategy(torch.optim.SGD(model.parameters(), lr=0.1))

    with pytest.raises(ValueError, match="checkpoint optimizer"):
        sgd.load_state_dict(state)
