"""Shared-engine adapter for the bidirectional multitask protein critic."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import partial
from typing import Any

import torch
import torch.nn as nn

from src.training.contracts import (
    EngineState,
    MetricValue,
    StepContext,
    StepOutput,
    TrainingCheckpoint,
    TrainingPhase,
)


def task_losses(
    logits_dict: dict,
    batch: dict,
    classification_tasks: tuple[str, ...],
    regression_tasks: tuple[str, ...],
    criterion: nn.Module | dict[str, nn.Module],
) -> dict[str, torch.Tensor]:
    """Calculate available single-label and regression objectives."""
    losses = {}
    for task in classification_tasks:
        targets = batch[task]
        valid = targets != -1
        if bool(valid.any()):
            task_criterion = criterion[task] if isinstance(criterion, dict) else criterion
            losses[task] = task_criterion(logits_dict[task][valid], targets[valid])
    for task in regression_tasks:
        targets = batch[task].float()
        valid = torch.isfinite(targets)
        if bool(valid.any()):
            predictions = logits_dict[task].squeeze(-1)
            losses[task] = nn.functional.smooth_l1_loss(
                predictions[valid], targets[valid]
            )
    return losses


class ProteinCriticTask:
    """Protein classification/regression objectives behind a generic task contract."""

    def __init__(
        self,
        *,
        model: nn.Module,
        train_loader,
        validation_loader,
        device: torch.device,
        classification_tasks: Sequence[str],
        regression_tasks: Sequence[str],
        multi_label_tasks: Sequence[str],
        train_classification_criteria: Mapping[str, nn.Module],
        validation_classification_criterion: nn.Module,
        multi_label_criteria: Mapping[str, nn.Module],
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.device = device
        self.classification_tasks = tuple(classification_tasks)
        self.regression_tasks = tuple(regression_tasks)
        self.multi_label_tasks = tuple(multi_label_tasks)
        self.train_classification_criteria = dict(train_classification_criteria)
        self.validation_classification_criterion = validation_classification_criterion
        self.multi_label_criteria = dict(multi_label_criteria)
        self._phase_totals: dict[str, float] = {}
        self._phase_weights: dict[str, float] = {}

    def begin_phase(self, phase: TrainingPhase, epoch: int) -> None:
        self._phase_totals = {}
        self._phase_weights = {}
        if phase == TrainingPhase.TRAIN:
            sampler = getattr(self.train_loader, "batch_sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            self.model.train()
        else:
            self.model.eval()

    def end_phase(self, phase: TrainingPhase, epoch: int):
        metrics = {
            name: MetricValue(total / self._phase_weights[name])
            for name, total in self._phase_totals.items()
            if self._phase_weights.get(name, 0.0) > 0
        }
        if self.device.type == "mps":
            torch.mps.empty_cache()
        return metrics

    def train_batches(self, epoch: int):
        return self.train_loader

    def validation_batches(self, epoch: int):
        return self.validation_loader

    def training_step(self, batch, context: StepContext) -> StepOutput:
        return self._step(batch, training=True)

    def validation_step(self, batch, context: StepContext) -> StepOutput:
        return self._step(batch, training=False)

    def _step(self, batch, *, training: bool) -> StepOutput:
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        logits = self.model(input_ids, attention_mask=attention_mask)
        criteria = (
            self.train_classification_criteria
            if training
            else self.validation_classification_criterion
        )
        supervised = task_losses(
            logits,
            {
                task: batch[task].to(self.device)
                for task in (*self.classification_tasks, *self.regression_tasks)
            },
            self.classification_tasks,
            self.regression_tasks,
            criteria,
        )
        components: list[torch.Tensor] = []
        metrics: dict[str, MetricValue] = {}
        if supervised:
            components.append(torch.stack(list(supervised.values())).mean())
            for name, value in supervised.items():
                metrics[f"{name}_loss"] = MetricValue(float(value.detach()))
        for task in self.classification_tasks:
            targets = batch[task].to(self.device)
            valid = targets != -1
            if bool(valid.any()):
                correct = int((logits[task][valid].argmax(dim=-1) == targets[valid]).sum().item())
                self._record(f"{task}_accuracy", correct, int(valid.sum().item()))
        for task in self.regression_tasks:
            targets = batch[task].to(self.device).float()
            valid = torch.isfinite(targets)
            if bool(valid.any()):
                error = (logits[task].squeeze(-1)[valid] - targets[valid]).abs().sum()
                self._record(f"{task}_mae", float(error.detach()), int(valid.sum().item()))

        for task in self.multi_label_tasks:
            targets = batch[task].to(self.device)
            if targets.numel() and bool((targets >= 0).any()):
                value = self.multi_label_criteria[task](logits[task], targets)
                components.append(value)
                metrics[f"{task}_loss"] = MetricValue(float(value.detach()))
                valid = targets >= 0
                predictions = logits[task] >= 0
                correct = (predictions[valid] == targets[valid].bool()).sum()
                self._record(
                    f"{task}_accuracy", float(correct.detach()), int(valid.sum().item())
                )

        if components:
            loss = torch.stack(components).sum()
        else:
            # Keep accumulation boundaries deterministic even for an unlabeled batch.
            loss = sum((value.sum() * 0.0 for value in logits.values()))

        loss_value = float(loss.detach())
        loss_weight = 1.0 if training or components else 0.0
        metrics["loss"] = MetricValue(loss_value, loss_weight)
        sequences = int(input_ids.shape[0])
        residues = int(attention_mask.sum().item()) if attention_mask is not None else int(input_ids.numel())
        for name, value in metrics.items():
            self._phase_totals[name] = self._phase_totals.get(name, 0.0) + value.total
            self._phase_weights[name] = self._phase_weights.get(name, 0.0) + value.weight
        return StepOutput(
            loss=loss,
            metrics=metrics,
            committed_units={"sequences": sequences, "residues": residues},
        )

    def _record(self, name: str, total: float, weight: float) -> None:
        self._phase_totals[name] = self._phase_totals.get(name, 0.0) + float(total)
        self._phase_weights[name] = self._phase_weights.get(name, 0.0) + float(weight)

    def state_dict(self) -> Mapping[str, Any]:
        return {"model": self.model.state_dict()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.model.load_state_dict(state["model"])


def decode_protein_critic_checkpoint(payload: Mapping[str, Any]) -> TrainingCheckpoint:
    """Read current engine checkpoints or the legacy critic schema."""
    if "training_contract_version" in payload:
        return TrainingCheckpoint.from_payload(payload)
    required = {"model_state_dict", "optimizer_state_dict", "epoch"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"legacy protein critic checkpoint is missing: {sorted(missing)}")
    epoch = int(payload["epoch"])
    complete = bool(payload.get("epoch_complete", True))
    completed = epoch + 1 if complete else epoch
    return TrainingCheckpoint(
        engine=EngineState(
            completed_epochs=completed,
            current_epoch=completed if complete else epoch,
            microbatch=0 if complete else int(payload.get("microbatch_idx", 0)),
            optimizer_step=int(payload.get("optimizer_step", 0)),
        ),
        task={"model": payload["model_state_dict"]},
        strategy={"optimizer": payload["optimizer_state_dict"]},
        rng=payload.get("rng_state", {}),
        metadata={
            "best_metric": payload.get("best_val_loss"),
            "legacy": True,
        },
    )


def adapt_protein_critic_checkpoint(
    payload: dict[str, Any],
    *,
    config: Mapping[str, Any],
    dataset_provenance: Mapping[str, Any],
    task_vocabs: Mapping[str, Any],
    model_spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Add evaluator-compatible legacy aliases to an engine checkpoint."""
    engine = payload["engine"]
    reason = payload["metadata"]["reason"]
    complete = reason in {"epoch", "epoch_archive", "best", "best_archive"}
    best_metric = payload["metadata"].get("best_metric")
    payload.update(
        {
            "epoch": int(engine["completed_epochs"]) - 1 if complete else int(engine["current_epoch"]),
            "epoch_complete": complete,
            "microbatch_idx": 0 if complete else int(engine["microbatch"]),
            "optimizer_step": int(engine["optimizer_step"]),
            "model_state_dict": payload["task"]["model"],
            "optimizer_state_dict": payload["strategy"]["optimizer"],
            "best_val_loss": float("inf") if best_metric is None else best_metric,
            "checkpoint_reason": "best_epoch" if reason in {"best", "best_archive"} else reason,
            "cfg": dict(config),
            "dataset_provenance": dict(dataset_provenance),
            "task_vocabs": dict(task_vocabs),
            "model_spec": dict(model_spec),
            "rng_state": payload["rng"],
        }
    )
    return payload


def make_protein_critic_checkpoint_adapter(**kwargs):
    return partial(adapt_protein_critic_checkpoint, **kwargs)
