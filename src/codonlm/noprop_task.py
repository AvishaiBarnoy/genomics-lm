"""Shared-engine adapters for layer-local NoProp training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F

from src.training.contracts import (
    EngineState,
    MetricValue,
    StepContext,
    StepOutput,
    TrainingCheckpoint,
    TrainingPhase,
    UpdateResult,
)
from src.training.strategies import NonFiniteStepError


class NoPropTask:
    """Data, model, and local denoising objectives used by NoProp."""

    def __init__(
        self,
        *,
        model,
        train_loader,
        validation_loader,
        device: torch.device,
        noise_sigma: float,
        train_generator: torch.Generator | None = None,
        seed: int = 0,
    ) -> None:
        if noise_sigma < 0:
            raise ValueError("noise_sigma must be non-negative")
        self.model = model
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.device = device
        self.noise_sigma = float(noise_sigma)
        self.train_generator = train_generator
        self.seed = int(seed)
        self._train_totals: dict[str, float] = {}
        self._train_steps = 0

    def begin_phase(self, phase: TrainingPhase, epoch: int) -> None:
        if phase == TrainingPhase.TRAIN:
            self._train_totals = {}
            self._train_steps = 0
            if self.train_generator is not None:
                self.train_generator.manual_seed(self.seed + epoch)
            self.model.train()
        else:
            self.model.eval()

    def end_phase(self, phase: TrainingPhase, epoch: int):
        if phase == TrainingPhase.TRAIN and self._train_steps:
            return {
                name: MetricValue(total / self._train_steps)
                for name, total in self._train_totals.items()
            }
        return {}

    def train_batches(self, epoch: int):
        return self.train_loader

    def validation_batches(self, epoch: int):
        return self.validation_loader

    def training_step(self, batch, context: StepContext) -> StepOutput:
        raise RuntimeError("NoProp training requires NoPropUpdateStrategy")

    def local_training_step(
        self,
        batch,
        *,
        embedding_optimizer: torch.optim.Optimizer,
        block_optimizers: Sequence[torch.optim.Optimizer],
        head_optimizer: torch.optim.Optimizer,
    ) -> StepOutput:
        x, y = (tensor.to(self.device) for tensor in batch)
        y_clean = self.model.tok_emb(y).detach()
        y_noisy = y_clean + torch.randn_like(y_clean) * self.noise_sigma
        non_pad_mask = y.ne(0).unsqueeze(-1).float()
        h, attn_mask = self._initial_hidden(x)

        embedding_optimizer.zero_grad(set_to_none=True)
        block_losses: list[torch.Tensor] = []
        for index, (block, optimizer) in enumerate(
            zip(self.model.blocks, block_optimizers, strict=True)
        ):
            h_in = h if index == 0 else h.detach()
            optimizer.zero_grad(set_to_none=True)
            h, pred_y = block(h_in, noisy_targets=y_noisy, attn_mask=attn_mask)
            loss = self._masked_mse(pred_y, y_clean, non_pad_mask)
            self._require_finite(loss, f"block {index} loss")
            loss.backward()
            optimizer.step()
            if index == 0:
                embedding_optimizer.step()
            block_losses.append(loss.detach())

        head_optimizer.zero_grad(set_to_none=True)
        logits = self.model.head(self.model.ln_f(h.detach()))
        ce = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=0)
        self._require_finite(ce, "head loss")
        ce.backward()
        head_optimizer.step()
        output = self._output(ce.detach(), block_losses, y)
        for name, value in output.metrics.items():
            self._train_totals[name] = self._train_totals.get(name, 0.0) + value.total
        self._train_steps += 1
        return output

    def validation_step(self, batch, context: StepContext) -> StepOutput:
        x, y = (tensor.to(self.device) for tensor in batch)
        y_clean = self.model.tok_emb(y)
        non_pad_mask = y.ne(0).unsqueeze(-1).float()
        h, attn_mask = self._initial_hidden(x)
        block_losses = []
        for block in self.model.blocks:
            h, pred_y = block(h, noisy_targets=y_clean, attn_mask=attn_mask)
            block_losses.append(self._masked_mse(pred_y, y_clean, non_pad_mask))
        logits = self.model.head(self.model.ln_f(h))
        ce = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=0)
        return self._output(ce, block_losses, y)

    def _initial_hidden(self, x: torch.Tensor):
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        hidden = self.model.drop(self.model.tok_emb(x) + self.model.pos_emb(positions))
        attn_mask = None
        if self.model.sep_id is not None:
            segments = torch.cumsum(x.eq(int(self.model.sep_id)), dim=1)
            attn_mask = segments.unsqueeze(-1).eq(segments.unsqueeze(-2)).unsqueeze(1)
        return hidden, attn_mask

    @staticmethod
    def _masked_mse(prediction, target, non_pad_mask):
        elementwise = F.mse_loss(prediction, target, reduction="none")
        denominator = non_pad_mask.sum() * prediction.size(-1) + 1e-8
        return (elementwise * non_pad_mask).sum() / denominator

    @staticmethod
    def _require_finite(loss: torch.Tensor, name: str) -> None:
        if not bool(torch.isfinite(loss.detach()).item()):
            raise NonFiniteStepError(f"nonfinite NoProp {name}")

    @staticmethod
    def _output(ce, block_losses, targets):
        metrics = {"loss": MetricValue(float(ce.detach()))}
        metrics.update(
            {
                f"block_{index}_mse": MetricValue(float(loss.detach()))
                for index, loss in enumerate(block_losses)
            }
        )
        return StepOutput(
            loss=ce,
            metrics=metrics,
            committed_units={
                "tokens": int(targets.ne(0).sum().item()),
                "sequences": int(targets.size(0)),
            },
        )

    def state_dict(self) -> Mapping[str, Any]:
        return {"model": self.model.state_dict()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.model.load_state_dict(state["model"])


class NoPropUpdateStrategy:
    """Commit NoProp's embedding, block-local, and output-head updates."""

    def __init__(self, embedding_optimizer, block_optimizers, head_optimizer):
        self.embedding_optimizer = embedding_optimizer
        self.block_optimizers = list(block_optimizers)
        self.head_optimizer = head_optimizer
        self._active = False
        self._processed = 0

    def begin_group(self, group_size: int) -> None:
        if group_size != 1:
            raise ValueError("NoProp requires grad_accum_steps=1")
        if self._active:
            raise RuntimeError("a NoProp update group is already active")
        self._active = True
        self._processed = 0

    def process_microbatch(self, task, batch, context: StepContext) -> StepOutput:
        if not self._active:
            raise RuntimeError("begin_group must be called before process_microbatch")
        if not isinstance(task, NoPropTask):
            raise TypeError("NoPropUpdateStrategy requires NoPropTask")
        output = task.local_training_step(
            batch,
            embedding_optimizer=self.embedding_optimizer,
            block_optimizers=self.block_optimizers,
            head_optimizer=self.head_optimizer,
        )
        output.validate()
        self._processed = 1
        return output

    def commit_group(self) -> UpdateResult:
        if not self._active or self._processed != 1:
            raise RuntimeError("NoProp update group is incomplete")
        self._reset()
        return UpdateResult(committed=True, optimizer_steps=1)

    def abort_group(self, reason: str) -> UpdateResult:
        for optimizer in self._optimizers():
            optimizer.zero_grad(set_to_none=True)
        self._reset()
        return UpdateResult(committed=False, optimizer_steps=0, reason=reason)

    def end_epoch(self, metrics) -> None:
        return None

    def state_dict(self) -> Mapping[str, Any]:
        return {
            "optimizers": {
                "embedding": self.embedding_optimizer.state_dict(),
                "blocks": [optimizer.state_dict() for optimizer in self.block_optimizers],
                "head": self.head_optimizer.state_dict(),
            }
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        optimizers = state["optimizers"]
        block_states = optimizers["blocks"]
        if len(block_states) != len(self.block_optimizers):
            raise ValueError("checkpoint NoProp block optimizer count differs from model")
        self.embedding_optimizer.load_state_dict(optimizers["embedding"])
        for optimizer, saved in zip(self.block_optimizers, block_states, strict=True):
            optimizer.load_state_dict(saved)
        self.head_optimizer.load_state_dict(optimizers["head"])
        self._reset()

    def _optimizers(self):
        return [self.embedding_optimizer, *self.block_optimizers, self.head_optimizer]

    def _reset(self):
        self._active = False
        self._processed = 0


def decode_noprop_checkpoint(payload: Mapping[str, Any]) -> TrainingCheckpoint:
    """Read current engine checkpoints or the legacy NoProp schema."""
    if "training_contract_version" in payload:
        return TrainingCheckpoint.from_payload(payload)
    required = {"model", "optimizers", "epoch"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"legacy NoProp checkpoint is missing: {sorted(missing)}")
    epoch = int(payload["epoch"])
    return TrainingCheckpoint(
        engine=EngineState(
            completed_epochs=epoch,
            current_epoch=epoch,
            microbatch=0,
            optimizer_step=int(payload.get("run_progress", {}).get("optimizer_step", 0)),
        ),
        task={"model": payload["model"]},
        strategy={"optimizers": payload["optimizers"]},
        rng=payload.get("rng_state", {}),
        metadata={"best_metric": payload.get("best_val_loss"), "legacy": True},
    )


def adapt_noprop_checkpoint(payload: dict[str, Any]) -> dict[str, Any]:
    """Add legacy NoProp aliases to a versioned checkpoint payload."""
    engine = payload["engine"]
    metrics = payload["metadata"].get("metrics", {})
    payload.update(
        {
            "model": payload["task"]["model"],
            "optimizers": payload["strategy"]["optimizers"],
            "epoch": int(engine["completed_epochs"]),
            "epoch_complete": payload["metadata"]["reason"] in {"epoch", "best"},
            "val_loss": metrics.get("loss"),
            "best_val_loss": payload["metadata"].get("best_metric"),
            "rng_state": payload["rng"],
        }
    )
    return payload
