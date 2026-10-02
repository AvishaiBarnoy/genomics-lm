"""Small atomic session snapshots for inspection without loading checkpoints."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
import time
import uuid
import warnings


class TrainingStatus:
    """Record phase boundaries and at most one progress snapshot per 30 seconds.

    This is progress telemetry, not a background heartbeat: an unchanged timestamp
    does not prove a hang. A kill/power loss may leave a nonterminal snapshot.
    """

    def __init__(self, engine):
        self.engine = engine
        self.last_write = float("-inf")
        task = type(engine.task).__name__
        self.data = dict(
            schema_version=1,
            session_id=uuid.uuid4().hex,
            started_at=self._now(),
            pid=os.getpid(),
            launch_mode=engine.run.launch_mode,
            task=task,
            run_type={
                "CodonLMTask": "codonlm",
                "ProteinCriticTask": "protein-critic",
            }.get(task, "unknown"),
            target_epochs=engine.config.epochs,
            resume_checkpoint=str(engine.run.resume_checkpoint)
            if engine.run.resume_checkpoint
            else None,
            status="active",
        )

    @staticmethod
    def _now():
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def write(self, event, *, metrics=None, metadata=None, status="active", error=None):
        """Persist available numerical and lifecycle evidence; logging never masks training errors."""
        now = time.monotonic()
        if event == "group_committed" and now - self.last_write < 30:
            return
        engine = self.engine
        self.data.update(
            updated_at=self._now(),
            status=status,
            last_event=event,
            progress=asdict(engine.state),
            aborted_groups=engine.aborted_groups,
            best_metric=engine.best_metric,
            best_epoch=engine.best_epoch,
            committed_units=dict(engine.committed_units),
        )
        if metrics:
            self.data["latest_metrics"] = {
                name: value.total for name, value in metrics.items()
            }
            self.data["metrics_event"] = event
        if metadata:
            self.data["event_details"] = metadata
            if event == "checkpoint_saved":
                self.data["last_checkpoint"] = dict(metadata)
        else:
            self.data.pop("event_details", None)
        optimizer = getattr(engine.strategy, "optimizer", None)
        if optimizer is not None:
            self.data["learning_rates"] = [
                group["lr"] for group in optimizer.param_groups
            ]
        if error is not None:
            self.data["error"] = dict(type=type(error).__name__, message=str(error))
        if status != "active":
            self.data["finished_at"] = self._now()

        def serializable(value):
            if isinstance(value, float) and not math.isfinite(value):
                return str(value)
            if isinstance(value, dict):
                return {k: serializable(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [serializable(v) for v in value]
            return value

        path = engine.run.run_dir / "run_status.json"
        temporary = path.with_suffix(".json.tmp")
        try:
            temporary.write_text(
                json.dumps(
                    serializable(self.data), indent=2, sort_keys=True, allow_nan=False
                )
                + "\n"
            )
            os.replace(temporary, path)
            self.last_write = now
        except (OSError, TypeError, ValueError) as exc:
            warnings.warn(f"Could not write training status: {exc}", RuntimeWarning)
