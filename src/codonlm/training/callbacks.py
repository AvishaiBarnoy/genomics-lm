"""CodonLM training callbacks and presentation adapters."""

from __future__ import annotations

import csv
import math

from src.training.contracts import EngineEvent


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
            replay_metric = train.get("replay_term_loss")
            row.append("" if replay_metric is None else f"{replay_metric.total:.4f}")
        with self.curves_path.open("a", newline="") as handle:
            csv.writer(handle).writerow(row)
        print(
            f"[epoch {epoch}] train {train['loss'].total:.3f} | "
            f"val {validation['loss'].total:.3f} | next_val {val_next:.3f} | "
            f"ppl {perplexity:.2f}"
        )
