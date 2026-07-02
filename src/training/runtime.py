from __future__ import annotations

import os
import atexit
import faulthandler
import signal
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import torch


class WallTimeLimitException(Exception):
    """Raised when a trainer reaches its configured wall-time budget."""


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class WallTimer:
    max_minutes: float | None = None
    started_at: float = 0.0

    def __post_init__(self) -> None:
        self.started_at = time.perf_counter()

    @property
    def max_seconds(self) -> float | None:
        if self.max_minutes is None:
            return None
        return float(self.max_minutes) * 60.0

    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self.started_at

    def expired(self) -> bool:
        max_seconds = self.max_seconds
        return bool(max_seconds is not None and self.elapsed_seconds() > max_seconds)

    def check(self) -> None:
        if self.expired():
            raise WallTimeLimitException()


@dataclass
class PeriodicCheckpointPolicy:
    every_steps: int = 0
    every_minutes: float = 0.0
    last_saved_step: int = 0
    last_saved_at: float = 0.0

    def __post_init__(self) -> None:
        self.every_steps = int(self.every_steps or 0)
        self.every_minutes = float(self.every_minutes or 0.0)
        self.last_saved_at = time.perf_counter()

    def should_save(self, step: int) -> bool:
        if step <= self.last_saved_step:
            return False
        if self.every_steps > 0 and step % self.every_steps == 0:
            return True
        if self.every_minutes > 0:
            elapsed = time.perf_counter() - self.last_saved_at
            if elapsed >= self.every_minutes * 60.0:
                return True
        return False

    def mark_saved(self, step: int) -> None:
        self.last_saved_step = int(step)
        self.last_saved_at = time.perf_counter()


def save_checkpoint_atomic(payload: dict[str, Any], path: str | Path) -> None:
    """Write a torch checkpoint through a temporary file, then atomically replace."""
    final_path = Path(path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = final_path.with_name(f".{final_path.name}.tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, final_path)


class _Tee:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


class RunLogger:
    """Mirror stdout/stderr into a per-run log file."""

    def __init__(self, log_path: str | Path) -> None:
        self.log_path = Path(log_path)
        self._log_file: TextIO | None = None
        self._stdout: TextIO | None = None
        self._stderr: TextIO | None = None
        self._started_at: float | None = None
        self._closed = False
        self._old_threading_excepthook = None
        self._old_unraisablehook = None
        self._old_signal_handlers: dict[int, Any] = {}
        self._atexit_registered = False

    def __enter__(self) -> "RunLogger":
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = self.log_path.open("a", buffering=1)
        self._started_at = time.perf_counter()
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = _Tee(sys.stdout, self._log_file)  # type: ignore[assignment]
        sys.stderr = _Tee(sys.stderr, self._log_file)  # type: ignore[assignment]
        print(f"[log] writing run log to {self.log_path}")
        self._install_crash_hooks()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._write_exit_record(exc_type, exc, tb)
        self._restore_hooks()
        self._restore_streams()
        return False

    def _write_log_line(self, line: str) -> None:
        if self._log_file is None or self._closed:
            return
        self._log_file.write(line.rstrip("\n") + "\n")
        self._log_file.flush()

    def _write_exit_record(self, exc_type=None, exc=None, tb=None) -> None:
        if self._closed:
            return
        if exc_type is not None and self._log_file is not None:
            self._log_file.write("\n[error] unhandled exception:\n")
            traceback.print_exception(exc_type, exc, tb, file=self._log_file)
            self._log_file.flush()
        elapsed = None
        if self._started_at is not None:
            elapsed = time.perf_counter() - self._started_at
        status = "exception" if exc_type is not None else "exit"
        if elapsed is None:
            self._write_log_line(f"[log] run logger closing status={status}")
        else:
            self._write_log_line(f"[log] run logger closing status={status} elapsed_sec={elapsed:.2f}")

    def _restore_streams(self) -> None:
        if self._stdout is not None:
            sys.stdout = self._stdout
        if self._stderr is not None:
            sys.stderr = self._stderr
        if self._log_file is not None:
            self._log_file.close()
        self._closed = True

    def _install_crash_hooks(self) -> None:
        if self._log_file is None:
            return
        try:
            faulthandler.enable(file=self._log_file, all_threads=True)
        except Exception:
            pass

        self._old_threading_excepthook = getattr(threading, "excepthook", None)

        def thread_hook(args):
            self._write_log_line("[error] unhandled thread exception:")
            if self._log_file is not None:
                traceback.print_exception(args.exc_type, args.exc_value, args.exc_traceback, file=self._log_file)
                self._log_file.flush()
            if self._old_threading_excepthook is not None:
                self._old_threading_excepthook(args)

        if self._old_threading_excepthook is not None:
            threading.excepthook = thread_hook

        self._old_unraisablehook = getattr(sys, "unraisablehook", None)

        def unraisable_hook(unraisable):
            self._write_log_line(f"[error] unraisable exception: {unraisable.err_msg}")
            if self._log_file is not None:
                traceback.print_exception(
                    unraisable.exc_type,
                    unraisable.exc_value,
                    unraisable.exc_traceback,
                    file=self._log_file,
                )
                self._log_file.flush()
            if self._old_unraisablehook is not None:
                self._old_unraisablehook(unraisable)

        if self._old_unraisablehook is not None:
            sys.unraisablehook = unraisable_hook

        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            self._install_signal_handler(sig)

        if not self._atexit_registered:
            atexit.register(self._atexit_log)
            self._atexit_registered = True

    def _install_signal_handler(self, sig: signal.Signals) -> None:
        try:
            old_handler = signal.getsignal(sig)
            self._old_signal_handlers[int(sig)] = old_handler

            def handler(signum, frame):
                self._write_log_line(f"[signal] received {signal.Signals(signum).name}; exiting")
                if frame is not None and self._log_file is not None:
                    traceback.print_stack(frame, file=self._log_file)
                    self._log_file.flush()
                previous = self._old_signal_handlers.get(signum)
                self._restore_hooks()
                self._restore_streams()
                if callable(previous):
                    previous(signum, frame)
                elif previous == signal.SIG_IGN:
                    return
                raise SystemExit(128 + signum)

            signal.signal(sig, handler)
        except Exception:
            pass

    def _restore_hooks(self) -> None:
        for signum, old_handler in self._old_signal_handlers.items():
            try:
                signal.signal(signum, old_handler)
            except Exception:
                pass
        self._old_signal_handlers.clear()
        if self._old_threading_excepthook is not None:
            threading.excepthook = self._old_threading_excepthook
        if self._old_unraisablehook is not None:
            sys.unraisablehook = self._old_unraisablehook

    def _atexit_log(self) -> None:
        if not self._closed:
            self._write_log_line("[log] process atexit reached before logger close")
