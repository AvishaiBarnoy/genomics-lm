"""Model-agnostic contracts for the shared training engine.

These interfaces describe orchestration boundaries only. They intentionally do
not import any CodonLM or ProteinLM module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator, Mapping, Protocol, TypeVar, runtime_checkable

import torch


TRAINING_CONTRACT_VERSION = 1

BatchT = TypeVar("BatchT")


class BatchStream(Protocol[BatchT]):
    """Re-iterable or single-pass batch source with a declared finite length."""

    def __iter__(self) -> Iterator[BatchT]: ...

    def __len__(self) -> int: ...


class TrainingPhase(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"


@dataclass(frozen=True)
class EngineState:
    """Serializable progress owned by the model-agnostic engine."""

    completed_epochs: int = 0
    current_epoch: int = 0
    microbatch: int = 0
    optimizer_step: int = 0

    def __post_init__(self) -> None:
        for name in (
            "completed_epochs",
            "current_epoch",
            "microbatch",
            "optimizer_step",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class TrainingCheckpoint:
    """Namespaced checkpoint state shared by all future engine tasks."""

    engine: EngineState
    task: Mapping[str, Any]
    strategy: Mapping[str, Any]
    rng: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    contract_version: int = TRAINING_CONTRACT_VERSION

    def to_payload(self) -> dict[str, Any]:
        return {
            "training_contract_version": self.contract_version,
            "engine": {
                "completed_epochs": self.engine.completed_epochs,
                "current_epoch": self.engine.current_epoch,
                "microbatch": self.engine.microbatch,
                "optimizer_step": self.engine.optimizer_step,
            },
            "task": dict(self.task),
            "strategy": dict(self.strategy),
            "rng": dict(self.rng),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "TrainingCheckpoint":
        version = payload.get("training_contract_version")
        if version != TRAINING_CONTRACT_VERSION:
            raise ValueError(
                "unsupported training checkpoint contract version: "
                f"{version!r}; expected {TRAINING_CONTRACT_VERSION}"
            )
        required = {"engine", "task", "strategy", "rng"}
        missing = required.difference(payload)
        if missing:
            raise ValueError(f"training checkpoint is missing: {sorted(missing)}")
        engine = payload["engine"]
        return cls(
            engine=EngineState(
                completed_epochs=int(engine["completed_epochs"]),
                current_epoch=int(engine["current_epoch"]),
                microbatch=int(engine["microbatch"]),
                optimizer_step=int(engine["optimizer_step"]),
            ),
            task=payload["task"],
            strategy=payload["strategy"],
            rng=payload["rng"],
            metadata=payload.get("metadata", {}),
            contract_version=int(version),
        )


@dataclass(frozen=True)
class StepContext:
    """Engine-owned coordinates supplied to tasks and update strategies."""

    phase: TrainingPhase
    epoch: int
    microbatch: int
    optimizer_step: int
    device: torch.device
    group_size: int = 1

    def __post_init__(self) -> None:
        for name in ("epoch", "microbatch", "optimizer_step"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.group_size < 1:
            raise ValueError("group_size must be positive")


@dataclass(frozen=True)
class MetricValue:
    """A reducible metric numerator and its aggregation weight."""

    total: float
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.weight < 0:
            raise ValueError("metric weight must be non-negative")


@dataclass
class StepOutput:
    """Task output consumed by an update strategy or validation reducer."""

    loss: torch.Tensor
    metrics: Mapping[str, MetricValue] = field(default_factory=dict)
    committed_units: Mapping[str, int] = field(default_factory=dict)

    def validate(self) -> None:
        if self.loss.numel() != 1:
            raise ValueError("step loss must be a scalar tensor")
        for name, count in self.committed_units.items():
            if count < 0:
                raise ValueError(f"committed unit {name!r} must be non-negative")


@dataclass(frozen=True)
class UpdateResult:
    """Outcome of committing or aborting an accumulation group."""

    committed: bool
    optimizer_steps: int
    metrics: Mapping[str, MetricValue] = field(default_factory=dict)
    committed_units: Mapping[str, int] = field(default_factory=dict)
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.optimizer_steps < 0:
            raise ValueError("optimizer_steps must be non-negative")
        if not self.committed and self.optimizer_steps:
            raise ValueError("an aborted update cannot report optimizer steps")


@runtime_checkable
class TrainingTask(Protocol[BatchT]):
    """Model/data/objective adapter used by a training engine."""

    def begin_phase(self, phase: TrainingPhase, epoch: int) -> None: ...

    def train_batches(self, epoch: int) -> BatchStream[BatchT]: ...

    def validation_batches(self, epoch: int) -> BatchStream[BatchT]: ...

    def training_step(self, batch: BatchT, context: StepContext) -> StepOutput: ...

    def validation_step(self, batch: BatchT, context: StepContext) -> StepOutput: ...

    def state_dict(self) -> Mapping[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...


@runtime_checkable
class UpdateStrategy(Protocol[BatchT]):
    """Parameter-update algorithm, independent of engine iteration."""

    def begin_group(self, group_size: int) -> None: ...

    def process_microbatch(
        self,
        task: TrainingTask[BatchT],
        batch: BatchT,
        context: StepContext,
    ) -> StepOutput: ...

    def commit_group(self) -> UpdateResult: ...

    def abort_group(self, reason: str) -> UpdateResult: ...

    def state_dict(self) -> Mapping[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...


@dataclass(frozen=True)
class EngineEvent:
    """Stable callback payload that contains no model-specific assumptions."""

    name: str
    context: StepContext | None = None
    metrics: Mapping[str, MetricValue] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class TrainingCallback(Protocol):
    def on_event(self, event: EngineEvent) -> None: ...
