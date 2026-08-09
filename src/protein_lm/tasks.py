"""Training-task adapters for protein models."""

from __future__ import annotations

from collections.abc import Mapping
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


class ProteinLMTask:
    """Causal amino-acid language-model objective and data adapter."""

    def __init__(
        self,
        *,
        model: nn.Module,
        train_loader,
        validation_loader,
        tokenizer,
        device: torch.device,
        train_generator: torch.Generator,
        seed: int,
        log_every_microbatches: int = 100,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.tokenizer = tokenizer
        self.device = device
        self.train_generator = train_generator
        self.seed = int(seed)
        self.log_every_microbatches = int(log_every_microbatches)
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    def begin_phase(self, phase: TrainingPhase, epoch: int) -> None:
        if phase == TrainingPhase.TRAIN:
            self.train_generator.manual_seed(self.seed + epoch)
            self.model.train()
        else:
            self.model.eval()

    def end_phase(self, phase: TrainingPhase, epoch: int):
        return {}

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
                f"Loss: {output.loss.item():.4f}"
            )
        return output

    def validation_step(self, batch, context: StepContext) -> StepOutput:
        return self._step(batch)

    def _step(self, batch) -> StepOutput:
        input_ids = batch.to(self.device)
        targets = input_ids[:, 1:].contiguous()
        logits = self.model(input_ids[:, :-1]).contiguous()
        loss = self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        tokens = int(targets.ne(self.tokenizer.pad_token_id).sum().item())
        return StepOutput(
            loss=loss,
            metrics={"loss": MetricValue(float(loss.detach()), 1.0)},
            committed_units={"tokens": tokens, "sequences": int(input_ids.size(0))},
        )

    def state_dict(self) -> Mapping[str, Any]:
        return {"model": self.model.state_dict()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.model.load_state_dict(state["model"])


def decode_protein_lm_checkpoint(payload: Mapping[str, Any]) -> TrainingCheckpoint:
    """Read current engine checkpoints or the legacy ProteinLM schema."""

    if "training_contract_version" in payload:
        return TrainingCheckpoint.from_payload(payload)
    required = {"model_state_dict", "optimizer_state_dict", "epoch"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"legacy ProteinLM checkpoint is missing: {sorted(missing)}")
    epoch = int(payload["epoch"])
    complete = bool(payload.get("epoch_complete", True))
    completed_epochs = epoch + 1 if complete else epoch
    current_epoch = completed_epochs if complete else epoch
    strategy = {"optimizer": payload["optimizer_state_dict"]}
    if "scheduler_state_dict" in payload:
        strategy["scheduler"] = payload["scheduler_state_dict"]
    return TrainingCheckpoint(
        engine=EngineState(
            completed_epochs=completed_epochs,
            current_epoch=current_epoch,
            microbatch=0 if complete else int(payload.get("microbatch_idx", 0)),
            optimizer_step=int(payload.get("optimizer_step", 0)),
        ),
        task={"model": payload["model_state_dict"]},
        strategy=strategy,
        rng=payload.get("rng_state", {}),
        metadata={"best_metric": None, "legacy": True},
    )


def adapt_protein_lm_checkpoint(
    payload: dict[str, Any], *, config: Mapping[str, Any]
) -> dict[str, Any]:
    """Add legacy ProteinLM aliases to a versioned checkpoint payload."""

    engine = payload["engine"]
    reason = payload["metadata"]["reason"]
    complete = reason in {"epoch", "epoch_archive", "best"}
    payload.update(
        {
            "epoch": (
                int(engine["completed_epochs"]) - 1
                if complete
                else int(engine["current_epoch"])
            ),
            "epoch_complete": complete,
            "microbatch_idx": 0 if complete else int(engine["microbatch"]),
            "optimizer_step": int(engine["optimizer_step"]),
            "model_state_dict": payload["task"]["model"],
            "optimizer_state_dict": payload["strategy"]["optimizer"],
            "scheduler_state_dict": payload["strategy"].get("scheduler"),
            "checkpoint_reason": reason,
            "cfg": dict(config),
            "rng_state": payload["rng"],
        }
    )
    return payload


def make_protein_lm_checkpoint_adapter(config: Mapping[str, Any]):
    """Bind a run configuration to the engine's one-argument adapter callback."""

    return partial(adapt_protein_lm_checkpoint, config=config)
