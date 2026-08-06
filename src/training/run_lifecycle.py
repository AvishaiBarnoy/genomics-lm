from __future__ import annotations

import atexit
import csv
import hashlib
import fcntl
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


class RunLifecycleError(RuntimeError):
    """Raised when a launch would corrupt or ambiguously extend a training run."""


@dataclass(frozen=True)
class RunProgress:
    completed_epochs: int
    current_epoch: int
    microbatch: int
    optimizer_step: int


DEFAULT_MUTABLE_CONFIG_KEYS = {
    "checkpoint_every_minutes",
    "checkpoint_every_steps",
    "epochs",
    "log_every_steps",
    "max_time_minutes",
    "run_id",
}


def configuration_fingerprint(
    config: dict[str, Any], mutable_keys: set[str] | None = None
) -> str:
    excluded = DEFAULT_MUTABLE_CONFIG_KEYS if mutable_keys is None else mutable_keys
    def remove_mutable(value):
        if isinstance(value, dict):
            return {
                key: remove_mutable(item)
                for key, item in value.items()
                if key not in excluded
            }
        if isinstance(value, list):
            return [remove_mutable(item) for item in value]
        return value

    immutable = remove_mutable(config)
    encoded = json.dumps(immutable, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()


def checkpoint_progress(payload: dict[str, Any]) -> RunProgress:
    progress = payload.get("run_progress")
    if not isinstance(progress, dict):
        if "epoch" in payload and "epoch_complete" in payload:
            epoch = int(payload["epoch"])
            complete = bool(payload["epoch_complete"])
            return RunProgress(
                completed_epochs=epoch + 1 if complete else epoch,
                current_epoch=epoch + 1,
                microbatch=0 if complete else int(payload.get("microbatch_idx", 0)),
                optimizer_step=int(payload.get("optimizer_step", 0)),
            )
        raise RunLifecycleError(
            "Checkpoint has no unambiguous run_progress metadata. Legacy checkpoints "
            "must be migrated explicitly before in-place resume."
        )
    return RunProgress(
        completed_epochs=int(progress.get("completed_epochs", 0)),
        current_epoch=int(progress.get("current_epoch", 0)),
        microbatch=int(progress.get("microbatch", 0)),
        optimizer_step=int(progress.get("optimizer_step", 0)),
    )


def capture_rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": numpy_state[1].tolist(),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    get_mps_state = getattr(torch.mps, "get_rng_state", None)
    if torch.backends.mps.is_available() and callable(get_mps_state):
        state["torch_mps"] = get_mps_state()
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["bit_generator"],
            np.asarray(numpy_state["state"], dtype=np.uint32),
            numpy_state["position"],
            numpy_state["has_gauss"],
            numpy_state["cached_gaussian"],
        )
    )
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    set_mps_state = getattr(torch.mps, "set_rng_state", None)
    if "torch_mps" in state and callable(set_mps_state):
        set_mps_state(state["torch_mps"])


class TrainingRun:
    """Own a collision-safe training directory for one process."""

    def __init__(self, run_dir: Path, resume_checkpoint: Path | None) -> None:
        self.run_dir = run_dir
        self.resume_checkpoint = resume_checkpoint
        self.checkpoints = run_dir / "checkpoints"
        self.scores = run_dir / "scores"
        self.logs = run_dir / "logs"
        self.completion_path = run_dir / "run_complete.json"
        self.lock_path = run_dir / ".run.lock"
        self._lock_fd: int | None = None
        for path in (self.checkpoints, self.scores, self.logs):
            path.mkdir(parents=True, exist_ok=True)
        self._acquire_lock()
        atexit.register(self.close)

    @classmethod
    def open(
        cls,
        root: str | Path,
        run_id: str,
        *,
        resume: str | Path | None = None,
        last_checkpoint_name: str = "last.pt",
        target_epochs: int | None = None,
        curve_filename: str = "curves.csv",
        config_fingerprint: str | None = None,
    ) -> "TrainingRun":
        root = Path(root)
        if resume is None:
            run_dir = cls._allocate_serial(root, run_id)
            return cls(run_dir, None)

        checkpoint = Path(resume).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint}")
        run_dir = checkpoint.parent.parent if checkpoint.parent.name == "checkpoints" else checkpoint.parent
        if run_dir.name != run_id:
            raise RunLifecycleError(
                f"Resume checkpoint belongs to run '{run_dir.name}', but run ID "
                f"'{run_id}' was requested. Omit the override for in-place resume or "
                "use an explicit new run ID to fork."
            )
        completion_path = run_dir / "run_complete.json"
        newest = run_dir / "checkpoints" / last_checkpoint_name
        if not newest.is_file() or checkpoint != newest.resolve():
            raise RunLifecycleError(
                f"Cannot resume run '{run_id}' from {checkpoint.name}. Use the newest "
                f"{last_checkpoint_name} or provide a new run ID to fork."
            )
        payload = torch.load(checkpoint, map_location="cpu")
        progress = checkpoint_progress(payload)
        saved_fingerprint = payload.get("run_fingerprint")
        if (
            config_fingerprint is not None
            and saved_fingerprint is not None
            and config_fingerprint != saved_fingerprint
        ):
            raise RunLifecycleError(
                "Resume configuration changes immutable run settings. Use the "
                "checkpoint's configuration or a new run ID to fork."
            )
        cls._validate_curve_history(
            run_dir / "scores" / curve_filename, progress.completed_epochs
        )
        if target_epochs is not None and int(target_epochs) <= progress.completed_epochs:
            raise RunLifecycleError(
                f"Run has {progress.completed_epochs} completed epochs, but target "
                f"epochs is {target_epochs}. Set epochs greater than "
                f"{progress.completed_epochs} or use a new run ID."
            )
        if completion_path.exists() and target_epochs is None:
            raise RunLifecycleError(
                f"Run '{run_id}' is complete. Specify a greater total epoch target "
                "or use a new run ID."
            )
        run = cls(run_dir, checkpoint)
        if completion_path.exists():
            archived = run_dir / f"run_complete_epoch_{progress.completed_epochs:03d}.json"
            os.replace(completion_path, archived)
        return run

    @staticmethod
    def _validate_curve_history(path: Path, completed_epochs: int) -> None:
        if not path.exists():
            return
        with path.open(newline="") as handle:
            rows = list(csv.reader(handle))
        epochs = []
        for row in rows[1:]:
            if not row:
                continue
            try:
                epochs.append(int(row[0]))
            except ValueError as exc:
                raise RunLifecycleError(
                    f"Invalid epoch value in curve history: {row[0]!r}"
                ) from exc
        if epochs != sorted(set(epochs)):
            raise RunLifecycleError(
                f"Curve history contains duplicate or decreasing epochs: {path}"
            )
        if epochs and epochs[-1] > completed_epochs:
            raise RunLifecycleError(
                f"Curve history reaches epoch {epochs[-1]}, but the selected last "
                f"checkpoint has only {completed_epochs} completed epochs. Use a new "
                "run ID or repair the run explicitly."
            )

    @staticmethod
    def _allocate_serial(root: Path, run_id: str) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        for serial in range(1, 10000):
            name = run_id if serial == 1 else f"{run_id}-r{serial:03d}"
            candidate = root / name
            try:
                candidate.mkdir()
                return candidate
            except FileExistsError:
                continue
        raise RunLifecycleError(f"Could not allocate a serial directory for {run_id}")

    def _acquire_lock(self) -> None:
        self._lock_fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(self._lock_fd)
            self._lock_fd = None
            raise RunLifecycleError(
                f"Run directory is already locked: {self.run_dir}"
            ) from exc
        os.ftruncate(self._lock_fd, 0)
        os.write(self._lock_fd, f"pid={os.getpid()}\n".encode())

    def mark_complete(self, metadata: dict[str, Any]) -> None:
        payload = {"status": "complete", **metadata}
        temporary = self.completion_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, self.completion_path)

    def logger(self, filename: str = "train.log"):
        from src.training.runtime import RunLogger

        return RunLogger(self.logs / filename)

    def close(self) -> None:
        if self._lock_fd is None:
            return
        fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
        os.close(self._lock_fd)
        self._lock_fd = None

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> "TrainingRun":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.close()
        return False
