"""Shared-engine adapter for single-label protein classification."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import Any

import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from src.training.contracts import (
    EngineState,
    MetricValue,
    StepContext,
    StepOutput,
    TrainingCheckpoint,
    TrainingPhase,
)


class ProteinClassifierTask:
    """Single-label protein objective, metrics, data order, and model state."""

    def __init__(
        self,
        *,
        model: nn.Module,
        train_loader,
        validation_loader,
        criterion: nn.Module,
        device: torch.device,
        train_generator: torch.Generator,
        seed: int,
        label_map: Mapping[str, int],
        pad_token_id: int,
        log_every_microbatches: int = 100,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.criterion = criterion
        self.device = device
        self.train_generator = train_generator
        self.seed = int(seed)
        self.label_map = dict(label_map)
        self.pad_token_id = int(pad_token_id)
        self.log_every_microbatches = int(log_every_microbatches)
        self._predictions: list[int] = []
        self._labels: list[int] = []

    def begin_phase(self, phase: TrainingPhase, epoch: int) -> None:
        self._predictions = []
        self._labels = []
        if phase == TrainingPhase.TRAIN:
            self.train_generator.manual_seed(self.seed + epoch)
            self.model.train()
        else:
            self.model.eval()

    def end_phase(self, phase: TrainingPhase, epoch: int):
        if not self._labels:
            return {}
        correct = sum(
            prediction == label
            for prediction, label in zip(self._predictions, self._labels)
        )
        return {
            "accuracy": MetricValue(correct / len(self._labels)),
            "weighted_f1": MetricValue(
                f1_score(
                    self._labels,
                    self._predictions,
                    average="weighted",
                    zero_division=0,
                )
            ),
        }

    def train_batches(self, epoch: int):
        return self.train_loader

    def validation_batches(self, epoch: int):
        return self.validation_loader

    def training_step(self, batch, context: StepContext) -> StepOutput:
        output = self._step(batch)
        if (
            self.log_every_microbatches > 0
            and context.microbatch % self.log_every_microbatches == 0
        ):
            print(
                f"Epoch {context.epoch + 1}, Step {context.microbatch}, "
                f"Loss: {output.loss.item():.4f}",
                flush=True,
            )
        return output

    def validation_step(self, batch, context: StepContext) -> StepOutput:
        return self._step(batch)

    def _step(self, batch) -> StepOutput:
        input_ids, labels = batch
        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)
        attention_mask = input_ids.ne(self.pad_token_id)
        logits = self.model(input_ids, attention_mask=attention_mask)
        loss = self.criterion(logits, labels)
        predictions = logits.argmax(dim=-1)
        self._predictions.extend(predictions.detach().cpu().tolist())
        self._labels.extend(labels.detach().cpu().tolist())
        return StepOutput(
            loss=loss,
            metrics={"loss": MetricValue(float(loss.detach()))},
            committed_units={
                "sequences": int(input_ids.shape[0]),
                "residues": int(attention_mask.sum().item()),
            },
        )

    def state_dict(self) -> Mapping[str, Any]:
        return {"model": self.model.state_dict(), "label_map": self.label_map}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        saved_label_map = state.get("label_map")
        checkpoint_label_map = (
            None if saved_label_map is None else dict(saved_label_map)
        )
        if checkpoint_label_map is not None and checkpoint_label_map != self.label_map:
            raise ValueError("Resume checkpoint label map differs from the training data")
        self.model.load_state_dict(state["model"])


def decode_protein_classifier_checkpoint(
    payload: Mapping[str, Any],
) -> TrainingCheckpoint:
    """Read a current engine checkpoint or the legacy classifier schema."""
    if "training_contract_version" in payload:
        return TrainingCheckpoint.from_payload(payload)
    required = {
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "epoch",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"legacy protein classifier checkpoint is missing: {sorted(missing)}")
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
        task={
            "model": payload["model_state_dict"],
            "label_map": payload.get("label_map"),
        },
        strategy={
            "optimizer": payload["optimizer_state_dict"],
            "scheduler": payload["scheduler_state_dict"],
        },
        rng=payload.get("rng_state", {}),
        metadata={"best_metric": None, "legacy": True},
    )


def adapt_protein_classifier_checkpoint(
    payload: dict[str, Any], *, config: Mapping[str, Any]
) -> dict[str, Any]:
    """Add legacy classifier aliases to a versioned engine checkpoint."""
    engine = payload["engine"]
    reason = payload["metadata"]["reason"]
    complete = reason in {"epoch", "epoch_archive", "best", "best_archive"}
    payload.update(
        {
            "epoch": int(engine["completed_epochs"]) - 1 if complete else int(engine["current_epoch"]),
            "epoch_complete": complete,
            "microbatch_idx": 0 if complete else int(engine["microbatch"]),
            "optimizer_step": int(engine["optimizer_step"]),
            "model_state_dict": payload["task"]["model"],
            "optimizer_state_dict": payload["strategy"]["optimizer"],
            "scheduler_state_dict": payload["strategy"]["scheduler"],
            "label_map": dict(payload["task"]["label_map"]),
            "loss": payload["metadata"].get("metrics", {}).get(
                "loss", float("inf")
            ),
            "checkpoint_reason": reason,
            "cfg": dict(config),
            "rng_state": payload["rng"],
        }
    )
    return payload


def make_protein_classifier_checkpoint_adapter(config: Mapping[str, Any]):
    return partial(adapt_protein_classifier_checkpoint, config=config)
