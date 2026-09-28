"""Shared-engine task for independent classifiers over frozen features."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from src.training.contracts import MetricValue, StepContext, StepOutput, TrainingPhase


@dataclass(frozen=True)
class HeadBatch:
    task_name: str
    features: torch.Tensor
    labels: torch.Tensor


class HeadBatchStream:
    """Present several finite head-specific loaders as one engine batch stream."""

    def __init__(self, loaders: Mapping[str, Any]) -> None:
        self.loaders = dict(loaders)

    def __len__(self) -> int:
        return sum(len(loader) for loader in self.loaders.values())

    def __iter__(self):
        for task_name, loader in self.loaders.items():
            for batch in loader:
                yield HeadBatch(task_name, batch["X"], batch["label"])


class MLPHeadsTask:
    """Independent cross-entropy objectives sharing a checkpointed head container."""

    def __init__(
        self,
        *,
        model: nn.Module,
        train_loaders: Mapping[str, Any],
        validation_loaders: Mapping[str, Any],
        device: torch.device,
        train_generators: Mapping[str, torch.Generator],
        seed: int,
        task_dims: Mapping[str, int],
    ) -> None:
        self.model = model
        self.train_loaders = dict(train_loaders)
        self.validation_loaders = dict(validation_loaders)
        self.device = device
        self.train_generators = dict(train_generators)
        self.seed = int(seed)
        self.task_dims = dict(task_dims)
        self.criterion = nn.CrossEntropyLoss()
        self.best_validation_losses = {
            name: float("inf") for name in self.task_dims
        }
        self.best_head_states: dict[str, Mapping[str, Any]] = {}
        self._loss_totals: dict[str, float] = {}
        self._loss_counts: dict[str, int] = {}

    def begin_phase(self, phase: TrainingPhase, epoch: int) -> None:
        self._loss_totals = {}
        self._loss_counts = {}
        if phase == TrainingPhase.TRAIN:
            for offset, generator in enumerate(self.train_generators.values()):
                generator.manual_seed(self.seed + epoch + offset * 1_000_003)
            self.model.train()
        else:
            self.model.eval()

    def end_phase(self, phase: TrainingPhase, epoch: int):
        metrics = {
            f"{name}_loss": MetricValue(
                self._loss_totals[name] / self._loss_counts[name]
            )
            for name in self._loss_totals
        }
        if phase == TrainingPhase.VALIDATION:
            for name in self.task_dims:
                loss = metrics[f"{name}_loss"].total
                if loss < self.best_validation_losses[name]:
                    self.best_validation_losses[name] = loss
                    self.best_head_states[name] = copy.deepcopy(
                        self.model.heads[name].state_dict()
                    )
        return metrics

    def train_batches(self, epoch: int):
        return HeadBatchStream(self.train_loaders)

    def validation_batches(self, epoch: int):
        return HeadBatchStream(self.validation_loaders)

    def training_step(self, batch: HeadBatch, context: StepContext) -> StepOutput:
        return self._step(batch)

    def validation_step(self, batch: HeadBatch, context: StepContext) -> StepOutput:
        return self._step(batch)

    def _step(self, batch: HeadBatch) -> StepOutput:
        features = batch.features.to(self.device)
        labels = batch.labels.to(self.device)
        logits = self.model.heads[batch.task_name](features)
        loss = self.criterion(logits, labels)
        detached_loss = float(loss.detach())
        self._loss_totals[batch.task_name] = (
            self._loss_totals.get(batch.task_name, 0.0) + detached_loss
        )
        self._loss_counts[batch.task_name] = self._loss_counts.get(batch.task_name, 0) + 1
        return StepOutput(
            loss=loss,
            metrics={"loss": MetricValue(detached_loss)},
            committed_units={"samples": int(labels.size(0))},
        )

    def restore_best_heads(self) -> None:
        missing = set(self.task_dims).difference(self.best_head_states)
        if missing:
            raise RuntimeError(f"no selected checkpoint for heads: {sorted(missing)}")
        for name, state in self.best_head_states.items():
            self.model.heads[name].load_state_dict(state)

    def state_dict(self) -> Mapping[str, Any]:
        return {
            "model": self.model.state_dict(),
            "task_dims": self.task_dims,
            "best_validation_losses": self.best_validation_losses,
            "best_head_states": self.best_head_states,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if dict(state["task_dims"]) != self.task_dims:
            raise ValueError("checkpoint task dimensions differ from configured vocabularies")
        self.model.load_state_dict(state["model"])
        self.best_validation_losses = {
            name: float(value)
            for name, value in state.get("best_validation_losses", {}).items()
        }
        self.best_head_states = dict(state.get("best_head_states", {}))
