"""Model-agnostic parameter update strategies."""

from __future__ import annotations

import math
from collections.abc import Mapping
from contextlib import nullcontext
from typing import Any, Generic

import torch

from src.training.contracts import (
    BatchT,
    StepContext,
    StepOutput,
    TrainingTask,
    UpdateResult,
)


class NonFiniteStepError(RuntimeError):
    """Raised when a loss or accumulated gradient is not finite."""


class PrecisionPolicy:
    """Autocast and optional gradient scaling independent of model code."""

    def __init__(
        self,
        *,
        device_type: str = "cpu",
        dtype: torch.dtype | None = None,
        enabled: bool = False,
        scale_gradients: bool = False,
    ) -> None:
        self.device_type = device_type
        self.dtype = dtype
        self.enabled = bool(enabled)
        self.scaler = torch.amp.GradScaler(
            device_type, enabled=self.enabled and scale_gradients
        )

    def forward_context(self):
        if not self.enabled:
            return nullcontext()
        return torch.amp.autocast(
            device_type=self.device_type, dtype=self.dtype, enabled=True
        )

    def backward(self, loss: torch.Tensor) -> None:
        self.scaler.scale(loss).backward()

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        self.scaler.unscale_(optimizer)

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        self.scaler.step(optimizer)
        self.scaler.update()

    def state_dict(self) -> Mapping[str, Any]:
        return self.scaler.state_dict()

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.scaler.load_state_dict(dict(state))


class AccumulatedBackpropStrategy(Generic[BatchT]):
    """Standard backpropagation with actual-size gradient averaging."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        scheduler: Any | None = None,
        parameters=None,
        grad_clip_norm: float | None = None,
        precision: PrecisionPolicy | None = None,
        scheduler_interval: str = "update",
        scheduler_metric: str = "loss",
    ) -> None:
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.parameters = list(parameters) if parameters is not None else [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        self.grad_clip_norm = grad_clip_norm
        self.precision = precision or PrecisionPolicy()
        if scheduler_interval not in {"update", "epoch"}:
            raise ValueError("scheduler_interval must be 'update' or 'epoch'")
        self.scheduler_interval = scheduler_interval
        self.scheduler_metric = scheduler_metric
        self._expected_group_size = 0
        self._processed = 0

    def begin_group(self, group_size: int) -> None:
        if group_size < 1:
            raise ValueError("group_size must be positive")
        if self._processed:
            raise RuntimeError("cannot begin a group while another group is active")
        self._expected_group_size = int(group_size)
        self.optimizer.zero_grad(set_to_none=True)

    def process_microbatch(
        self,
        task: TrainingTask[BatchT],
        batch: BatchT,
        context: StepContext,
    ) -> StepOutput:
        if self._expected_group_size < 1:
            raise RuntimeError("begin_group must be called before process_microbatch")
        with self.precision.forward_context():
            output = task.training_step(batch, context)
        output.validate()
        if not bool(torch.isfinite(output.loss.detach()).item()):
            raise NonFiniteStepError("nonfinite microbatch loss")
        self.precision.backward(output.loss)
        self._processed += 1
        return output

    def commit_group(self) -> UpdateResult:
        if self._processed != self._expected_group_size:
            raise RuntimeError(
                f"cannot commit {self._processed} microbatches; "
                f"expected {self._expected_group_size}"
            )
        self.precision.unscale_(self.optimizer)
        scale = 1.0 / self._processed
        for parameter in self.parameters:
            if parameter.grad is not None:
                parameter.grad.mul_(scale)
        finite = all(
            parameter.grad is None
            or bool(torch.isfinite(parameter.grad).all().item())
            for parameter in self.parameters
        )
        if not finite:
            return self.abort_group("nonfinite accumulated gradient")
        if self.grad_clip_norm is not None:
            norm = torch.nn.utils.clip_grad_norm_(self.parameters, self.grad_clip_norm)
            if not math.isfinite(float(norm)):
                return self.abort_group("nonfinite clipped gradient norm")
        self.precision.step(self.optimizer)
        if self.scheduler is not None and self.scheduler_interval == "update":
            self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self._reset_group()
        return UpdateResult(committed=True, optimizer_steps=1)

    def abort_group(self, reason: str) -> UpdateResult:
        self.optimizer.zero_grad(set_to_none=True)
        self._reset_group()
        return UpdateResult(committed=False, optimizer_steps=0, reason=reason)

    def end_epoch(self, metrics: Mapping[str, Any]) -> None:
        if self.scheduler is None or self.scheduler_interval != "epoch":
            return
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            metric = metrics.get(self.scheduler_metric)
            if metric is None:
                raise ValueError(
                    f"epoch scheduler requires metric {self.scheduler_metric!r}"
                )
            self.scheduler.step(metric.total)
        else:
            self.scheduler.step()

    def _reset_group(self) -> None:
        self._expected_group_size = 0
        self._processed = 0

    def state_dict(self) -> Mapping[str, Any]:
        optimizer_type = (
            f"{type(self.optimizer).__module__}.{type(self.optimizer).__qualname__}"
        )
        state: dict[str, Any] = {
            "optimizer": self.optimizer.state_dict(),
            "optimizer_type": optimizer_type,
        }
        state["precision"] = dict(self.precision.state_dict())
        state["scheduler_interval"] = self.scheduler_interval
        if self.scheduler is not None:
            state["scheduler"] = self.scheduler.state_dict()
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        saved_optimizer_type = state.get("optimizer_type")
        current_optimizer_type = (
            f"{type(self.optimizer).__module__}.{type(self.optimizer).__qualname__}"
        )
        if (
            saved_optimizer_type is not None
            and saved_optimizer_type != current_optimizer_type
        ):
            raise ValueError(
                f"checkpoint optimizer {saved_optimizer_type!r} differs from "
                f"configured optimizer {current_optimizer_type!r}"
            )
        self.optimizer.load_state_dict(state["optimizer"])
        saved_interval = state.get("scheduler_interval", self.scheduler_interval)
        if saved_interval != self.scheduler_interval:
            raise ValueError(
                f"checkpoint scheduler interval {saved_interval!r} differs from "
                f"configured interval {self.scheduler_interval!r}"
            )
        self.precision.load_state_dict(state.get("precision", {}))
        if self.scheduler is not None:
            if "scheduler" not in state:
                raise ValueError("checkpoint has no scheduler state")
            self.scheduler.load_state_dict(state["scheduler"])
        self._reset_group()
