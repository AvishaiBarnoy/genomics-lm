"""Shared-engine task for synthetic DNA-shape encoder pretraining."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from src.training.contracts import MetricValue, StepContext, StepOutput, TrainingPhase


class BiophysicsEncoderTask:
    """Regress codon-aligned DNA-shape targets from nucleotide one-hot inputs."""

    def __init__(
        self,
        *,
        model: nn.Module,
        train_loader,
        validation_loader,
        device: torch.device,
        train_generator: torch.Generator,
        seed: int,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.device = device
        self.train_generator = train_generator
        self.seed = int(seed)
        self.criterion = nn.MSELoss()
        self.best_validation_loss = float("inf")
        self.best_model_state: Mapping[str, Any] | None = None
        self._phase_loss = 0.0
        self._phase_weight = 0

    def begin_phase(self, phase: TrainingPhase, epoch: int) -> None:
        self._phase_loss = 0.0
        self._phase_weight = 0
        if phase == TrainingPhase.TRAIN:
            self.train_generator.manual_seed(self.seed + epoch)
            self.model.train()
        else:
            self.model.eval()

    def end_phase(self, phase: TrainingPhase, epoch: int):
        if self._phase_weight == 0:
            return {}
        loss = self._phase_loss / self._phase_weight
        if phase == TrainingPhase.VALIDATION and loss < self.best_validation_loss:
            self.best_validation_loss = loss
            self.best_model_state = copy.deepcopy(self.model.state_dict())
        return {"loss": MetricValue(loss)}

    def train_batches(self, epoch: int):
        return self.train_loader

    def validation_batches(self, epoch: int):
        return self.validation_loader

    def training_step(self, batch, context: StepContext) -> StepOutput:
        return self._step(batch)

    def validation_step(self, batch, context: StepContext) -> StepOutput:
        return self._step(batch)

    def _step(self, batch) -> StepOutput:
        one_hot, targets = (tensor.to(self.device) for tensor in batch)
        loss = self.criterion(self.model(one_hot), targets)
        detached_loss = float(loss.detach())
        batch_weight = int(one_hot.size(0))
        self._phase_loss += detached_loss * batch_weight
        self._phase_weight += batch_weight
        return StepOutput(
            loss=loss,
            metrics={"loss": MetricValue(detached_loss)},
            committed_units={
                "sequences": int(one_hot.size(0)),
                "nucleotides": int(one_hot.shape[0] * one_hot.shape[1]),
            },
        )

    def restore_best_model(self) -> None:
        if self.best_model_state is None:
            raise RuntimeError("no validation-selected encoder state is available")
        self.model.load_state_dict(self.best_model_state)

    def state_dict(self) -> Mapping[str, Any]:
        return {
            "model": self.model.state_dict(),
            "best_validation_loss": self.best_validation_loss,
            "best_model_state": self.best_model_state,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.model.load_state_dict(state["model"])
        self.best_validation_loss = float(
            state.get("best_validation_loss", float("inf"))
        )
        self.best_model_state = state.get("best_model_state")
