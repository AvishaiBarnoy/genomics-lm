"""Shared-engine adapter for latent protein energy-model training."""

from __future__ import annotations

import random
from collections.abc import Mapping
from functools import partial
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.contracts import (
    EngineState,
    MetricValue,
    StepContext,
    StepOutput,
    TrainingCheckpoint,
    TrainingPhase,
)


AMINO_ACIDS = tuple("ARNDCQEGHILKMFPSTWYV")


def corrupt_sequence(seq: str, mutation_rate: float = 0.20) -> str:
    """Create a substitution decoy using the process-global restored RNG."""
    if not seq:
        raise ValueError("cannot corrupt an empty protein sequence")
    if not 0.0 < mutation_rate <= 1.0:
        raise ValueError("mutation_rate must be in (0, 1]")
    residues = list(seq)
    count = max(1, int(len(seq) * mutation_rate))
    for index in random.sample(range(len(seq)), count):
        residues[index] = random.choice(AMINO_ACIDS)
    return "".join(residues)


class ProteinEBMTask:
    """Real-versus-corrupted latent ranking with a frozen critic backbone."""

    def __init__(
        self,
        *,
        model: nn.Module,
        critic: nn.Module,
        train_loader,
        validation_loader,
        tokenizer,
        device: torch.device,
        block_size: int,
        mutation_rate: float = 0.20,
        log_every_microbatches: int = 50,
    ) -> None:
        self.model = model
        self.critic = critic
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.tokenizer = tokenizer
        self.device = device
        self.block_size = int(block_size)
        self.mutation_rate = float(mutation_rate)
        self.log_every_microbatches = int(log_every_microbatches)
        if any(parameter.requires_grad for parameter in critic.parameters()):
            raise ValueError("ProteinEBM critic parameters must be frozen")
        self._phase_totals: dict[str, float] = {}
        self._phase_weights: dict[str, float] = {}

    def begin_phase(self, phase: TrainingPhase, epoch: int) -> None:
        self._phase_totals = {}
        self._phase_weights = {}
        self.critic.eval()
        self.model.train(phase == TrainingPhase.TRAIN)

    def end_phase(self, phase: TrainingPhase, epoch: int):
        return {
            name: MetricValue(total / self._phase_weights[name])
            for name, total in self._phase_totals.items()
            if self._phase_weights.get(name, 0.0) > 0
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
                f"Epoch {context.epoch + 1} | Step {context.microbatch} | "
                f"Loss: {output.loss.item():.4f} | "
                f"E_pos: {output.metrics['energy_pos'].total:.3f} | "
                f"E_neg: {output.metrics['energy_neg'].total:.3f}",
                flush=True,
            )
        return output

    def validation_step(self, batch, context: StepContext) -> StepOutput:
        return self._step(batch)

    def _step(self, batch) -> StepOutput:
        pos_ids = batch["input_ids"].to(self.device)
        pos_mask = batch["attention_mask"].to(self.device)
        neg_ids, neg_mask = self._negative_batch(
            batch["sequence"], target_length=pos_ids.shape[1]
        )
        with torch.no_grad():
            positive_latent = self.critic.extract_latent(pos_ids, pos_mask)
            negative_latent = self.critic.extract_latent(neg_ids, neg_mask)
        energy_pos = self.model(positive_latent)
        energy_neg = self.model(negative_latent)
        loss = F.softplus(energy_pos - energy_neg).mean()
        batch_size = int(pos_ids.shape[0])
        metrics = {
            "loss": MetricValue(float(loss.detach())),
            "energy_pos": MetricValue(float(energy_pos.detach().mean())),
            "energy_neg": MetricValue(float(energy_neg.detach().mean())),
            "energy_gap": MetricValue(
                float((energy_neg.detach() - energy_pos.detach()).mean())
            ),
        }
        for name, value in metrics.items():
            self._phase_totals[name] = self._phase_totals.get(name, 0.0) + value.total
            self._phase_weights[name] = self._phase_weights.get(name, 0.0) + value.weight
        return StepOutput(
            loss=loss,
            metrics=metrics,
            committed_units={
                "sequences": batch_size,
                "residues": int(pos_mask.sum().item()),
            },
        )

    def _negative_batch(self, sequences, *, target_length: int):
        ids = []
        masks = []
        for sequence in sequences:
            negative = corrupt_sequence(sequence, self.mutation_rate)
            tokens = (
                [self.tokenizer.bos_token_id]
                + self.tokenizer.encode_sequence(negative)[: self.block_size - 2]
                + [self.tokenizer.eos_token_id]
            )
            tokens = tokens[:target_length]
            padding = max(0, target_length - len(tokens))
            ids.append(tokens + [self.tokenizer.pad_token_id] * padding)
            masks.append([1] * len(tokens) + [0] * padding)
        return (
            torch.tensor(ids, dtype=torch.long, device=self.device),
            torch.tensor(masks, dtype=torch.long, device=self.device),
        )

    def state_dict(self) -> Mapping[str, Any]:
        return {"model": self.model.state_dict()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.model.load_state_dict(state["model"])


def decode_protein_ebm_checkpoint(payload: Mapping[str, Any]) -> TrainingCheckpoint:
    """Read a versioned checkpoint or the legacy one-based EBM schema."""
    if "training_contract_version" in payload:
        return TrainingCheckpoint.from_payload(payload)
    required = {"model", "optimizer_state_dict", "epoch"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"legacy ProteinEBM checkpoint is missing: {sorted(missing)}")
    epoch = int(payload["epoch"])
    complete = bool(payload.get("epoch_complete", True))
    progress = payload.get("run_progress", {})
    return TrainingCheckpoint(
        engine=EngineState(
            completed_epochs=epoch if complete else max(0, epoch - 1),
            current_epoch=epoch if complete else max(0, epoch - 1),
            microbatch=0 if complete else int(payload.get("microbatch_idx", 0)),
            optimizer_step=int(progress.get("optimizer_step", 0)),
        ),
        task={"model": payload["model"]},
        strategy={"optimizer": payload["optimizer_state_dict"]},
        rng=payload.get("rng_state", {}),
        metadata={
            "best_metric": payload.get("best_val_loss"),
            "best_epoch": payload.get("best_epoch"),
            "legacy": True,
        },
    )


def adapt_protein_ebm_checkpoint(
    payload: dict[str, Any],
    *,
    critic_checkpoint: str,
    model_spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Add the established EBM aliases to a versioned checkpoint."""
    engine = payload["engine"]
    metadata = payload["metadata"]
    reason = metadata["reason"]
    complete = reason in {"epoch", "epoch_archive", "best", "best_archive"}
    epoch = (
        int(engine["completed_epochs"])
        if complete
        else int(engine["current_epoch"]) + 1
    )
    current_loss = metadata.get("metrics", {}).get("loss", float("inf"))
    best_metric = metadata.get("best_metric")
    payload.update(
        {
            "model": payload["task"]["model"],
            "epoch": epoch,
            "val_loss": current_loss,
            "optimizer_state_dict": payload["strategy"]["optimizer"],
            "best_val_loss": float("inf") if best_metric is None else best_metric,
            "best_epoch": metadata.get("best_epoch") or 0,
            "epoch_complete": complete,
            "microbatch_idx": 0 if complete else int(engine["microbatch"]),
            "checkpoint_reason": reason,
            "rng_state": payload["rng"],
            "critic_checkpoint": critic_checkpoint,
            "model_spec": dict(model_spec),
        }
    )
    return payload


def make_protein_ebm_checkpoint_adapter(**kwargs):
    return partial(adapt_protein_ebm_checkpoint, **kwargs)
