"""CodonLM adapters for the shared model-agnostic training engine."""

from __future__ import annotations

import math
import resource
import csv
from collections.abc import Callable, Mapping
from functools import partial
from typing import Any

import torch

from src.codonlm.training.objectives import (
    multi_offset_lm_loss,
    termination_aux_loss,
    termination_distance_bucket_labels,
)
from src.training.contracts import (
    EngineEvent,
    EngineState,
    MetricValue,
    StepContext,
    StepOutput,
    TrainingCheckpoint,
    TrainingPhase,
)


PAD_ID = 0


class CodonLMConsole:
    """Write the historical curves schema and concise epoch telemetry."""

    def __init__(
        self,
        curves_path,
        *,
        multi_offset_weights,
        termination_enabled: bool,
        replay_enabled: bool,
        optimizer,
    ) -> None:
        self.curves_path = curves_path
        self.multi_offset_weights = dict(multi_offset_weights)
        self.termination_enabled = termination_enabled
        self.replay_enabled = replay_enabled
        self.optimizer = optimizer

    def on_event(self, event: EngineEvent) -> None:
        if event.name != "epoch_completed":
            return
        epoch = int(event.metadata["epoch"])
        train = event.metadata["training_metrics"]
        validation = event.metrics
        val_next = validation.get("next_loss", validation["loss"]).total
        perplexity = math.exp(min(20.0, val_next))
        row = [
            epoch,
            f"{train['loss'].total:.4f}",
            f"{validation['loss'].total:.4f}",
            f"{train.get('next_loss', train['loss']).total:.4f}",
            f"{val_next:.4f}",
            f"{perplexity:.3f}",
            f"{self.optimizer.param_groups[0]['lr']:.3e}",
        ]
        for offset in sorted(self.multi_offset_weights):
            row.extend(
                [
                    f"{train[f'offset_{offset}'].total:.4f}",
                    f"{validation[f'offset_{offset}'].total:.4f}",
                ]
            )
        if self.termination_enabled:
            row.extend(
                [
                    f"{train['term_loss'].total:.4f}",
                    f"{validation['term_loss'].total:.4f}",
                ]
            )
        if self.replay_enabled:
            row.append(f"{train['replay_term_loss'].total:.4f}")
        with self.curves_path.open("a", newline="") as handle:
            csv.writer(handle).writerow(row)
        print(
            f"[epoch {epoch}] train {train['loss'].total:.3f} | "
            f"val {validation['loss'].total:.3f} | next_val {val_next:.3f} | "
            f"ppl {perplexity:.2f}"
        )


class CodonLMTask:
    """Preserve CodonLM data, objective, and telemetry semantics behind an engine task."""

    def __init__(
        self,
        *,
        model,
        train_loader_factory: Callable[[int], Any],
        validation_loader,
        device: torch.device,
        config: Mapping[str, Any],
        encoder=None,
        lookup_table=None,
        replay_loader=None,
        termination_class_weights=None,
        replay_class_weights=None,
        multi_offset_weights=None,
    ) -> None:
        self.model = model
        self.train_loader_factory = train_loader_factory
        self.validation_loader = validation_loader
        self.device = device
        self.config = config
        self.encoder = encoder
        self.lookup_table = lookup_table
        self.replay_loader = replay_loader
        self.replay_iter = None
        self.termination_class_weights = termination_class_weights
        self.replay_class_weights = replay_class_weights
        self.multi_offset_weights = dict(multi_offset_weights or {})
        self.termination_loss_enabled = bool(config.get("termination_loss_enabled", False))
        self.replay_loss_enabled = bool(config.get("replay_loss_enabled", False))
        self.runtime_memory = {
            "process_max_rss_raw": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "mps_peak_allocated_bytes": 0,
            "mps_peak_driver_bytes": 0,
        }

    def begin_phase(self, phase: TrainingPhase, epoch: int) -> None:
        self.model.train(phase == TrainingPhase.TRAIN)
        if self.encoder is not None:
            self.encoder.train(
                phase == TrainingPhase.TRAIN
                and bool(self.config.get("unfreeze_encoder", False))
            )

    def end_phase(self, phase: TrainingPhase, epoch: int):
        return {}

    def train_batches(self, epoch: int):
        return self.train_loader_factory(epoch)

    def validation_batches(self, epoch: int):
        return self.validation_loader

    def training_step(self, batch, context: StepContext) -> StepOutput:
        return self._step(batch, context, training=True)

    def validation_step(self, batch, context: StepContext) -> StepOutput:
        return self._step(batch, context, training=False)

    def _shape_embeddings(self, tokens):
        if self.encoder is None or self.lookup_table is None:
            return None
        one_hots = self.lookup_table[tokens]
        one_hots = one_hots.view(tokens.size(0), 3 * tokens.size(1), 4)
        return self.encoder(one_hots)

    def _step(self, batch, context: StepContext, *, training: bool) -> StepOutput:
        xb, yb = (tensor.to(self.device) for tensor in batch)
        shapes = self._shape_embeddings(xb)
        need_aux = self.termination_loss_enabled or bool(self.multi_offset_weights)
        if need_aux:
            logits, next_loss, aux = self.model(
                xb, yb, return_aux=True, shape_embeddings=shapes
            )
        else:
            logits, next_loss = self.model(xb, yb, shape_embeddings=shapes)
            aux = {}
        loss = next_loss
        metrics = {"loss": MetricValue(float(loss.detach().cpu())),
                   "next_loss": MetricValue(float(next_loss.detach().cpu()))}

        if self.multi_offset_weights:
            offset_total, offset_losses = multi_offset_lm_loss(
                aux.get("offset_logits", logits),
                yb,
                self.multi_offset_weights,
                label_smoothing=float(self.config.get("label_smoothing", 0.0)),
                loss_weights=(
                    self.model.loss_weights
                    if not torch.all(self.model.loss_weights == 1.0).item()
                    else None
                ),
            )
            loss = loss + offset_total
            for offset, value in offset_losses.items():
                metrics[f"offset_{offset}"] = MetricValue(float(value.detach().cpu()))

        if self.termination_loss_enabled:
            term_logits = aux.get("termination_logits")
            if term_logits is None:
                raise RuntimeError(
                    "termination_loss_enabled=true but model returned no termination logits"
                )
            labels = termination_distance_bucket_labels(
                yb,
                stop_ids=tuple(int(x) for x in self.config.get("termination_stop_ids", [2])),
                bucket_edges=tuple(
                    int(x)
                    for x in self.config.get("termination_bucket_edges", [0, 3, 10, 30])
                ),
            )
            term_loss = termination_aux_loss(
                term_logits, labels, class_weights=self.termination_class_weights
            )
            loss = loss + float(self.config.get("termination_loss_weight", 0.1)) * term_loss
            metrics["term_loss"] = MetricValue(float(term_loss.detach().cpu()))

        if (
            training
            and self.replay_loader is not None
            and (context.microbatch + 1)
            % int(self.config.get("replay_every_microbatches", 1))
            == 0
        ):
            if self.replay_iter is None:
                self.replay_iter = iter(self.replay_loader)
            try:
                replay_x, replay_labels = next(self.replay_iter)
            except StopIteration:
                self.replay_iter = iter(self.replay_loader)
                replay_x, replay_labels = next(self.replay_iter)
            replay_x = replay_x.to(self.device)
            replay_labels = replay_labels.to(self.device)
            _, _, replay_aux = self.model(
                replay_x,
                return_aux=True,
                shape_embeddings=self._shape_embeddings(replay_x),
            )
            replay_logits = replay_aux.get("termination_logits")
            if replay_logits is None:
                raise RuntimeError(
                    "replay_loss_enabled=true but model returned no termination logits"
                )
            replay_loss = termination_aux_loss(
                replay_logits,
                replay_labels,
                class_weights=self.replay_class_weights,
            )
            loss = loss + float(self.config.get("replay_loss_weight", 0.1)) * replay_loss
            metrics["replay_term_loss"] = MetricValue(
                float(replay_loss.detach().cpu())
            )

        metrics["loss"] = MetricValue(float(loss.detach().cpu()))
        self._sample_runtime_memory()
        return StepOutput(
            loss=loss,
            metrics=metrics,
            committed_units={"tokens": int(yb.ne(PAD_ID).sum().item())},
        )

    def _sample_runtime_memory(self) -> None:
        self.runtime_memory["process_max_rss_raw"] = max(
            self.runtime_memory["process_max_rss_raw"],
            int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        )
        if self.device.type == "mps":
            self.runtime_memory["mps_peak_allocated_bytes"] = max(
                self.runtime_memory["mps_peak_allocated_bytes"],
                int(torch.mps.current_allocated_memory()),
            )
            driver = getattr(torch.mps, "driver_allocated_memory", None)
            if callable(driver):
                self.runtime_memory["mps_peak_driver_bytes"] = max(
                    self.runtime_memory["mps_peak_driver_bytes"], int(driver())
                )

    def state_dict(self):
        state = {
            "model": self.model.state_dict(),
            "runtime_memory": dict(self.runtime_memory),
        }
        if self.encoder is not None:
            state["encoder"] = self.encoder.state_dict()
        return state

    def load_state_dict(self, state) -> None:
        self.model.load_state_dict(state["model"])
        if self.encoder is not None and "encoder" in state:
            self.encoder.load_state_dict(state["encoder"])
        previous = state.get("runtime_memory", {})
        for key in self.runtime_memory:
            self.runtime_memory[key] = max(
                self.runtime_memory[key], int(previous.get(key, 0) or 0)
            )


def decode_codon_lm_checkpoint(payload: Mapping[str, Any]) -> TrainingCheckpoint:
    """Read versioned engine checkpoints and the legacy CodonLM schema."""
    if "training_contract_version" in payload:
        return TrainingCheckpoint.from_payload(payload)
    progress = payload.get("run_progress", {})
    completed = int(progress.get("completed_epochs", payload.get("epoch", 0)))
    current = int(progress.get("current_epoch", completed))
    strategy = {
        "optimizer": payload["optimizer"],
        "scheduler": payload.get("scheduler"),
        "scheduler_interval": (
            "update" if str(payload.get("cfg", {}).get("scheduler", "cosine")) == "cosine"
            else "epoch"
        ),
        "precision": {},
        "committed_steps": int(progress.get("optimizer_step", payload.get("step", 0))),
        "accumulation_health": payload.get("accumulation_health", {}),
    }
    return TrainingCheckpoint(
        engine=EngineState(
            completed_epochs=completed,
            current_epoch=current,
            microbatch=int(progress.get("microbatch", payload.get("epoch_microbatch_idx", 0))),
            optimizer_step=int(progress.get("optimizer_step", payload.get("step", 0))),
        ),
        task={
            "model": payload["model"],
            **({"encoder": payload["encoder"]} if "encoder" in payload else {}),
            "runtime_memory": payload.get("runtime_memory", {}),
        },
        strategy=strategy,
        rng=payload.get("rng_state", {}),
        metadata={
            "best_metric": payload.get("best_val"),
            "best_epoch": payload.get("best_epoch"),
            "committed_units": {"tokens": int(payload.get("consumed_train_tokens", 0))},
            "aborted_groups": int(
                payload.get("accumulation_health", {}).get("aborted_groups", 0)
            ),
            "no_improve": int(payload.get("no_improve", 0)),
            "legacy": True,
        },
    )


def adapt_codon_lm_checkpoint(
    payload: dict[str, Any],
    *,
    config: Mapping[str, Any],
    batch_size: int,
    grad_accum_steps: int,
    train_examples: int,
    train_batches: int,
    max_nonfinite_groups: int,
) -> dict[str, Any]:
    """Add the legacy CodonLM aliases consumed by analysis and resume tooling."""
    engine = payload["engine"]
    metadata = payload["metadata"]
    reason = metadata["reason"]
    complete = reason in {"epoch", "epoch_archive", "best"}
    validation = metadata.get("metrics", {})
    training = metadata.get("training_metrics", {})
    training_weights = metadata.get("training_metric_weights", {})
    active_totals = metadata.get("active_training_metric_totals", {})
    active_weights = metadata.get("active_training_metric_weights", {})
    if not training and active_weights:
        training = {
            name: float(total) / max(float(active_weights.get(name, 0)), 1.0)
            for name, total in active_totals.items()
        }
        training_weights = active_weights
    health = payload["strategy"].get("accumulation_health", {})
    val_loss = float(validation.get("loss", math.inf))
    payload.update(
        {
            "model": payload["task"]["model"],
            "optimizer": payload["strategy"]["optimizer"],
            "scheduler": payload["strategy"].get("scheduler"),
            "cfg": dict(config),
            "epoch": int(engine["completed_epochs"] if complete else engine["current_epoch"]),
            "val_loss": val_loss,
            "train_loss": training.get("loss"),
            "train_next_loss": training.get("next_loss"),
            "val_next_loss": validation.get("next_loss"),
            "train_term_loss": training.get("term_loss"),
            "val_term_loss": validation.get("term_loss"),
            "train_replay_term_loss": training.get("replay_term_loss"),
            "best_val": metadata.get("best_metric", math.inf),
            "best_epoch": metadata.get("best_epoch"),
            "no_improve": int(metadata.get("no_improve", 0)),
            "step": int(engine["optimizer_step"]),
            "consumed_train_tokens": int(metadata.get("committed_units", {}).get("tokens", 0)),
            "runtime_memory": dict(payload["task"].get("runtime_memory", {})),
            "epoch_microbatch_idx": 0 if complete else int(engine["microbatch"]),
            "last_seen_microbatch_idx": int(engine["microbatch"]),
            "batch_size": int(batch_size),
            "grad_accum_steps": int(grad_accum_steps),
            "train_examples": int(train_examples),
            "train_batches": int(train_batches),
            "accumulation_health": {
                "active_microbatches": 0,
                "nonfinite_microbatches": int(health.get("nonfinite_microbatches", 0)),
                "aborted_groups": int(health.get("aborted_groups", 0)),
                "discarded_finite_microbatches": int(
                    health.get("discarded_finite_microbatches", 0)
                ),
            },
            "max_nonfinite_accumulation_groups": int(max_nonfinite_groups),
            "epoch_train_metrics": {
                "total_loss_sum": float(training.get("loss", 0.0))
                * int(training_weights.get("loss", 0)),
                "next_loss_sum": float(training.get("next_loss", 0.0))
                * int(training_weights.get("next_loss", 0)),
                "microbatches": int(training_weights.get("loss", 0)),
                "initial_loss": training.get("loss"),
            },
            "rng_state": payload["rng"],
            "checkpoint_reason": reason,
        }
    )
    if "encoder" in payload["task"]:
        payload["encoder"] = payload["task"]["encoder"]
    return payload


def make_codon_lm_checkpoint_adapter(config: Mapping[str, Any], **facts):
    return partial(adapt_codon_lm_checkpoint, config=config, **facts)
