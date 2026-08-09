"""Model-agnostic training orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
from typing import Any, Callable, Generic, Mapping

import torch

from src.training.contracts import (
    BatchT,
    EngineEvent,
    EngineState,
    MetricValue,
    StepContext,
    TrainingCallback,
    TrainingCheckpoint,
    TrainingPhase,
    TrainingTask,
    UpdateStrategy,
)
from src.training.run_lifecycle import TrainingRun, capture_rng_state, restore_rng_state
from src.training.runtime import (
    PeriodicCheckpointPolicy,
    WallTimer,
    save_checkpoint_atomic,
)
from src.training.strategies import NonFiniteStepError


@dataclass(frozen=True)
class EngineConfig:
    epochs: int
    grad_accum_steps: int = 1
    validate_every_epochs: int = 1
    monitor: str = "loss"
    minimize_monitor: bool = True
    last_checkpoint_name: str = "last.pt"
    best_checkpoint_name: str = "best.pt"
    best_checkpoint_pattern: str | None = None
    epoch_checkpoint_pattern: str | None = None

    def __post_init__(self) -> None:
        for name in ("epochs", "grad_accum_steps", "validate_every_epochs"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class EngineResult:
    state: EngineState
    status: str
    best_metric: float | None
    aborted_groups: int = 0


@dataclass
class _MetricAccumulator:
    totals: dict[str, float] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)

    def add(self, metrics) -> None:
        for name, value in metrics.items():
            self.totals[name] = self.totals.get(name, 0.0) + float(value.total)
            self.weights[name] = self.weights.get(name, 0.0) + float(value.weight)

    def averages(self) -> dict[str, MetricValue]:
        return {
            name: MetricValue(total / self.weights[name], 1.0)
            for name, total in self.totals.items()
            if self.weights.get(name, 0.0) > 0
        }


class TrainingEngine(Generic[BatchT]):
    def __init__(
        self,
        *,
        task: TrainingTask[BatchT],
        strategy: UpdateStrategy[BatchT],
        run: TrainingRun,
        config: EngineConfig,
        device: torch.device,
        callbacks: list[TrainingCallback] | None = None,
        wall_timer: WallTimer | None = None,
        checkpoint_policy: PeriodicCheckpointPolicy | None = None,
        run_fingerprint: str | None = None,
        checkpoint_decoder: Callable[[Mapping[str, Any]], TrainingCheckpoint]
        | None = None,
        checkpoint_payload_adapter: Callable[[dict[str, Any]], dict[str, Any]]
        | None = None,
    ) -> None:
        self.task = task
        self.strategy = strategy
        self.run = run
        self.config = config
        self.device = device
        self.callbacks = list(callbacks or [])
        self.wall_timer = wall_timer or WallTimer()
        self.checkpoint_policy = checkpoint_policy or PeriodicCheckpointPolicy()
        self.run_fingerprint = run_fingerprint
        self.checkpoint_decoder = checkpoint_decoder or TrainingCheckpoint.from_payload
        self.checkpoint_payload_adapter = checkpoint_payload_adapter
        self.state = EngineState()
        self.best_metric: float | None = None
        self.aborted_groups = 0

    def fit(self) -> EngineResult:
        if self.run.resume_checkpoint is not None:
            self._restore(self.run.resume_checkpoint)

        for epoch in range(self.state.current_epoch, self.config.epochs):
            self.task.begin_phase(TrainingPhase.TRAIN, epoch)
            batches = self.task.train_batches(epoch)
            total_batches = len(batches)
            batch_iterator = iter(batches)
            resume_microbatch = self.state.microbatch if epoch == self.state.current_epoch else 0
            for _ in range(resume_microbatch):
                next(batch_iterator)

            microbatch = resume_microbatch
            while microbatch < total_batches:
                group_size = min(
                    self.config.grad_accum_steps, total_batches - microbatch
                )
                self.strategy.begin_group(group_size)
                group_metrics = _MetricAccumulator()
                group_units: dict[str, int] = {}
                group_failed = False
                for offset in range(group_size):
                    batch = next(batch_iterator)
                    context = StepContext(
                        TrainingPhase.TRAIN,
                        epoch,
                        microbatch + offset,
                        self.state.optimizer_step,
                        self.device,
                        group_size,
                    )
                    try:
                        output = self.strategy.process_microbatch(
                            self.task, batch, context
                        )
                    except NonFiniteStepError as exc:
                        self.strategy.abort_group(str(exc))
                        self.aborted_groups += 1
                        group_failed = True
                        for _ in range(offset + 1, group_size):
                            next(batch_iterator)
                        break
                    group_metrics.add(output.metrics)
                    for name, count in output.committed_units.items():
                        group_units[name] = group_units.get(name, 0) + int(count)

                microbatch += group_size
                if not group_failed:
                    update = self.strategy.commit_group()
                    if not update.committed:
                        self.aborted_groups += 1
                    else:
                        self.state = EngineState(
                            completed_epochs=epoch,
                            current_epoch=epoch,
                            microbatch=microbatch,
                            optimizer_step=(
                                self.state.optimizer_step + update.optimizer_steps
                            ),
                        )
                        self._emit(
                            "group_committed",
                            context,
                            group_metrics.averages(),
                            {"committed_units": group_units},
                        )
                        if self.checkpoint_policy.should_save(
                            self.state.optimizer_step
                        ):
                            self._save(self.config.last_checkpoint_name, "periodic")
                            self.checkpoint_policy.mark_saved(
                                self.state.optimizer_step
                            )
                if self.wall_timer.expired():
                    self.state = EngineState(
                        completed_epochs=epoch,
                        current_epoch=epoch,
                        microbatch=microbatch,
                        optimizer_step=self.state.optimizer_step,
                    )
                    self._save(self.config.last_checkpoint_name, "wall_time")
                    return EngineResult(
                        self.state, "interrupted", self.best_metric, self.aborted_groups
                    )

            training_metrics = dict(self.task.end_phase(TrainingPhase.TRAIN, epoch))
            self._emit("training_completed", None, training_metrics)
            validation_metrics = {}
            if (epoch + 1) % self.config.validate_every_epochs == 0:
                validation_metrics = self._validate(epoch)
            self.strategy.end_epoch(validation_metrics)
            self.state = EngineState(
                completed_epochs=epoch + 1,
                current_epoch=epoch + 1,
                microbatch=0,
                optimizer_step=self.state.optimizer_step,
            )
            monitored = validation_metrics.get(self.config.monitor)
            improved = monitored is not None and self._is_better(monitored.total)
            if improved:
                self.best_metric = monitored.total
            self._save(self.config.last_checkpoint_name, "epoch")
            if self.config.epoch_checkpoint_pattern is not None:
                self._save(
                    self.config.epoch_checkpoint_pattern.format(epoch=epoch + 1),
                    "epoch_archive",
                )
            if improved:
                self._save(self.config.best_checkpoint_name, "best")
                if self.config.best_checkpoint_pattern is not None:
                    self._save(
                        self.config.best_checkpoint_pattern.format(epoch=epoch + 1),
                        "best_archive",
                    )
            self._emit(
                "epoch_completed",
                None,
                validation_metrics,
                {
                    "epoch": epoch + 1,
                    "training_metrics": training_metrics,
                    "improved": improved,
                },
            )

        self.run.mark_complete(
            {
                "completed_epochs": self.state.completed_epochs,
                "optimizer_step": self.state.optimizer_step,
                "best_metric": self.best_metric,
            }
        )
        return EngineResult(self.state, "complete", self.best_metric, self.aborted_groups)

    def _validate(self, epoch: int) -> dict[str, MetricValue]:
        self.task.begin_phase(TrainingPhase.VALIDATION, epoch)
        accumulator = _MetricAccumulator()
        for microbatch, batch in enumerate(self.task.validation_batches(epoch)):
            context = StepContext(
                TrainingPhase.VALIDATION,
                epoch,
                microbatch,
                self.state.optimizer_step,
                self.device,
            )
            with torch.no_grad():
                output = self.task.validation_step(batch, context)
            output.validate()
            accumulator.add(output.metrics)
        metrics = accumulator.averages()
        metrics.update(self.task.end_phase(TrainingPhase.VALIDATION, epoch))
        self._emit("validation_completed", None, metrics)
        return metrics

    def _is_better(self, value: float) -> bool:
        if self.best_metric is None:
            return True
        return value < self.best_metric if self.config.minimize_monitor else value > self.best_metric

    def _save(self, filename: str, reason: str) -> None:
        checkpoint = TrainingCheckpoint(
            engine=self.state,
            task=self.task.state_dict(),
            strategy=self.strategy.state_dict(),
            rng=capture_rng_state(),
            metadata={"reason": reason, "best_metric": self.best_metric},
        )
        payload = checkpoint.to_payload()
        payload["run_progress"] = {
            "completed_epochs": self.state.completed_epochs,
            "current_epoch": self.state.current_epoch,
            "microbatch": self.state.microbatch,
            "optimizer_step": self.state.optimizer_step,
        }
        if self.run_fingerprint is not None:
            payload["run_fingerprint"] = self.run_fingerprint
        if self.checkpoint_payload_adapter is not None:
            payload = self.checkpoint_payload_adapter(payload)
        save_checkpoint_atomic(payload, self.run.checkpoints / filename)
        self._emit("checkpoint_saved", None, metadata={"reason": reason, "filename": filename})

    def _restore(self, path: Path) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        checkpoint = self.checkpoint_decoder(payload)
        self.task.load_state_dict(checkpoint.task)
        self.strategy.load_state_dict(checkpoint.strategy)
        restore_rng_state(dict(checkpoint.rng))
        self.state = checkpoint.engine
        best = checkpoint.metadata.get("best_metric")
        self.best_metric = None if best is None else float(best)

    def _emit(self, name, context=None, metrics=None, metadata=None) -> None:
        event = EngineEvent(name, context, metrics or {}, metadata or {})
        for callback in self.callbacks:
            callback.on_event(event)
