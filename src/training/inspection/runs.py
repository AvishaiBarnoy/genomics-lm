"""Bounded, read-only inspection of current and legacy training artifacts."""

from __future__ import annotations

import csv
import fcntl
import json
import math
import re
import time
from pathlib import Path

from .reporting import observed_at

MAX_BYTES = 2_000_000


def read_json(path: Path, warnings: list) -> dict | list | None:
    """Read bounded JSON; invalid or oversized artifacts remain visible as warnings."""
    try:
        if path.stat().st_size > MAX_BYTES:
            raise ValueError(f"exceeds {MAX_BYTES} byte inspection limit")
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        warnings.append(f"{path}: {exc}")
        return None


def lock_state(root: Path) -> str:
    """Probe the actual advisory lock without creating or truncating its file."""
    try:
        with (root / ".run.lock").open("rb") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return "held"
            fcntl.flock(handle, fcntl.LOCK_UN)
            return "released"
    except FileNotFoundError:
        return "absent"
    except OSError:
        return "unknown"


def identify_type(config: dict, model_spec: dict) -> str:
    """Identify supported families from explicit metadata, never directory names."""
    trainer = config.get("trainer")
    critic = (
        trainer in ("protein_multitask", "protein_critic")
        or bool(model_spec.get("task_dims"))
        or bool(config.get("task_vocabs"))
    )
    codon = (
        trainer == "codon_lm"
        or bool(config.get("train_npz"))
        or bool(
            config.get("data", {}).get("train_npz")
            if isinstance(config.get("data"), dict)
            else False
        )
    )
    if critic and codon:
        return "ambiguous"
    return "protein-critic" if critic else ("codonlm" if codon else "unknown")


def checkpoint_summary(path: Path, warnings: list) -> dict:
    """Load tensor-only checkpoints on CPU; never fall back to unsafe pickle loading."""
    info = dict(path=str(path), size_bytes=path.stat().st_size)
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError("checkpoint is not a mapping")
        config = payload.get("cfg", payload.get("config", {}))
        task = payload.get("task", {})
        if not config and isinstance(task, dict):
            config = task.get("cfg", task.get("config", {}))
        info.update(
            config=config if isinstance(config, dict) else {},
            model_spec=payload.get("model_spec", {}),
            progress=payload.get("run_progress", payload.get("engine", {})),
            metadata=payload.get("metadata", {}),
            training_contract_version=payload.get("training_contract_version"),
            best_epoch=payload.get("best_epoch"),
            accumulation_health=payload.get("accumulation_health", {}),
        )
        if (
            not info["progress"]
            and isinstance(payload.get("epoch"), int)
            and isinstance(payload.get("epoch_complete"), bool)
        ):
            epoch = payload["epoch"]
            complete = payload["epoch_complete"]
            info["progress"] = dict(
                completed_epochs=epoch + int(complete),
                current_epoch=epoch + 1,
                optimizer_step=payload.get("optimizer_step"),
                microbatch=payload.get("microbatch_idx"),
                source="legacy checkpoint; epoch field is zero-based",
            )
        if (
            not info["progress"]
            and isinstance(payload.get("epoch"), int)
            and isinstance(payload.get("step"), int)
            and isinstance(payload.get("epoch_microbatch_idx"), int)
        ):
            epoch = payload["epoch"]
            info["progress"] = dict(
                completed_epochs=epoch,
                current_epoch=epoch,
                optimizer_step=payload["step"],
                microbatch=payload["epoch_microbatch_idx"],
                source="legacy CodonLM checkpoint; epoch counts completed epochs",
            )
        if not info["metadata"]:
            info["metadata"] = {
                k: payload[k]
                for k in ["checkpoint_reason", "best_val_loss", "dataset_provenance"]
                if k in payload
            }
        if not info["accumulation_health"]:
            aborted = info["metadata"].get("aborted_groups")
            info["accumulation_health"] = (
                {"aborted_groups": aborted}
                if aborted is not None
                else {
                    "availability": "Not recorded by this checkpoint format; not evidence of zero errors."
                }
            )
    except Exception as exc:
        warnings.append(f"{path}: checkpoint metadata unavailable ({exc})")
    return info


def curve_summary(path: Path, warnings: list) -> dict:
    """Summarize completed numeric rows; ignore partial rows being written."""
    result = {"path": str(path)}
    try:
        if path.stat().st_size > MAX_BYTES:
            warnings.append(f"{path}: exceeds {MAX_BYTES} byte inspection limit")
            return result
        rows = []
        with path.open() as handle:
            for row in csv.DictReader(handle):
                if not row or None in row or any(v is None for v in row.values()):
                    warnings.append(f"{path}: incomplete CSV row ignored")
                    continue
                rows.append(row)
        if not rows:
            return result
        result.update(row_count=len(rows), latest=rows[-1])
        validation = []
        for row in rows:
            try:
                value = float(row.get("val_loss", ""))
                if math.isfinite(value):
                    validation.append((value, row))
                else:
                    warnings.append(
                        f"{path}: non-finite validation loss at epoch {row.get('epoch')}"
                    )
            except ValueError:
                continue
        if validation:
            result["best_validation"] = min(validation, key=lambda pair: pair[0])[1]
        if len(validation) > 1:
            result["validation_loss_change"] = validation[-1][0] - validation[-2][0]
    except (OSError, csv.Error, UnicodeError) as exc:
        warnings.append(f"{path}: {exc}")
    return result


def log_summary(path: Path) -> dict:
    """Inspect a bounded log tail, retaining evidence instead of inferring a crash."""
    with path.open("rb") as handle:
        size = path.stat().st_size
        handle.seek(max(0, size - MAX_BYTES))
        text = handle.read(MAX_BYTES).decode("utf-8", errors="replace")
    lines = text.replace("\r", "\n").splitlines()
    progress = [
        line
        for line in lines
        if re.search(r"\[progress\]|\[epoch \d+\]|Epoch \d+/", line)
    ]
    numerical = [
        line
        for line in lines
        if re.search(
            r"(?i)(?:loss|grad\w*)[=: ]+(?:nan|[+-]?inf)\b|non.finite|aborted.group",
            line,
        )
    ]
    errors = [
        line
        for line in lines
        if re.search(
            r"Traceback \(most recent|\[error\]|RuntimeError:|OutOfMemoryError:", line
        )
    ]
    latest = progress[-1] if progress else None
    telemetry = dict(
        re.findall(
            r"\b(epoch|step|optimizer_step|recent_loss|loss|lr|seq_per_sec|residues_per_sec)=([^\s]+)",
            next(
                (line for line in reversed(progress) if "[progress]" in line),
                latest or "",
            ),
        )
    )
    return dict(
        path=str(path),
        tail_truncated=size > MAX_BYTES,
        latest_progress=latest,
        telemetry=telemetry,
        recent_progress=progress[-3:],
        numerical_evidence=numerical[-5:],
        error_evidence=errors[-5:],
        tail=lines[-8:],
    )


def inspect_run(
    run_dir: Path,
    *,
    historical: bool = False,
    run_type: str = "auto",
    quiet_minutes: float = 30,
    now: float | None = None,
) -> dict:
    """Report evidence at observation time; no directory creation or trainer invocation."""
    root = run_dir.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Run directory does not exist: {root}")
    if quiet_minutes <= 0:
        raise ValueError("--quiet-minutes must be positive")
    now = time.time() if now is None else now
    warnings = []
    report = dict(
        schema_version=1,
        kind="run_analysis" if historical else "training_check",
        observed_at=observed_at(),
        run_dir=str(root),
        status="unknown",
        warnings=warnings,
    )
    lock = lock_state(root)
    report["lock"] = lock
    complete_path = root / "run_complete.json"
    completion = read_json(complete_path, warnings) if complete_path.exists() else None
    report["completion"] = completion
    status_path = root / "run_status.json"
    session = read_json(status_path, warnings) if status_path.exists() else None
    if isinstance(session, dict):
        report["session"] = session
    report["status"] = (
        "active"
        if lock == "held"
        else "complete"
        if lock in ("released", "absent")
        and isinstance(completion, dict)
        and completion.get("status") == "complete"
        else "incomplete"
        if lock == "released"
        else "unknown"
    )
    if lock == "held" and completion:
        warnings.append(
            "Completion marker coexists with an active lock; it does not establish completion of the active session."
        )
    config, spec = {}, {}
    if isinstance(session, dict):
        trainer = {"codonlm": "codon_lm", "protein-critic": "protein_multitask"}.get(
            session.get("run_type")
        )
        if trainer:
            config["trainer"] = trainer
    for path in sorted(root.glob("*.yaml")) + sorted(root.glob("*.yml")):
        if "config" not in path.stem:
            continue
        try:
            import yaml

            if path.stat().st_size > MAX_BYTES:
                raise ValueError("configuration exceeds inspection size limit")
            value = yaml.safe_load(path.read_text())
            if isinstance(value, dict):
                config.update(value)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            warnings.append(f"{path}: {exc}")
    for relative in ["meta.json", "checkpoints/meta.json", "config.json"]:
        path = root / relative
        if path.exists():
            data = read_json(path, warnings)
            if isinstance(data, dict):
                candidate = data.get("cfg", data.get("config", data))
                if isinstance(candidate, dict):
                    config.update(candidate)
                model_spec = data.get("model_spec")
                if isinstance(model_spec, dict):
                    spec.update(model_spec)
                elif model_spec is not None:
                    warnings.append(f"{path}: model_spec is not a mapping")
                report.setdefault("saved_training_state", {}).update(
                    {
                        k: data[k]
                        for k in [
                            "last_epoch",
                            "best_epoch",
                            "best_val_loss",
                            "last_train_loss",
                            "last_val_loss",
                            "accumulation_health",
                        ]
                        if k in data
                    }
                )
    checkpoints = sorted(
        set(root.glob("*.pt")) | set((root / "checkpoints").glob("*.pt"))
    )
    report["checkpoints"] = [
        dict(
            path=str(p),
            size_bytes=p.stat().st_size,
            age_seconds=round(max(0, now - p.stat().st_mtime), 1),
        )
        for p in checkpoints
    ]
    if historical and checkpoints:
        # Prefer latest resume state over a validation-selected best checkpoint.
        last = [p for p in checkpoints if "last" in p.stem]
        selected = max(
            last or checkpoints, key=lambda p: (p.stat().st_mtime_ns, str(p))
        )
        info = checkpoint_summary(selected, warnings)
        report["checkpoint_state"] = info
        config.update(info.pop("config", {}))
        spec.update(info.get("model_spec", {}))
        report["checkpoint_selection"] = (
            "Latest last checkpoint for progress; best filenames are listed separately. Actual evaluation selection is recorded in evaluation artifacts."
        )
    detected = identify_type(config, spec)
    report["run_type"] = detected if run_type == "auto" else run_type
    report["type_detection"] = dict(
        detected=detected, override=None if run_type == "auto" else run_type
    )
    if detected in ("unknown", "ambiguous"):
        warnings.append(
            "Run type cannot be established from available metadata; use --run-type if known."
        )
    if run_type != "auto" and detected not in ("unknown", run_type):
        warnings.append(
            f"Explicit run type {run_type} disagrees with detected type {detected}."
        )
    if historical:
        report["configuration"] = config
        lineage = root / "run_lineage.json"
        report["lineage"] = (
            read_json(lineage, warnings)
            if lineage.exists()
            else {
                "availability": "No fork lineage artifact recorded.",
                "meaning": "Lineage records the source run/checkpoint and configuration identity for an explicit full-state fork. Absence does not prove training started from scratch.",
            }
        )
        progress = report.get("checkpoint_state", {}).get("progress", {})
        completed, target = progress.get("completed_epochs"), config.get("epochs")
        report["completion_evidence"] = dict(
            marker_present=completion is not None,
            completed_epochs=completed,
            target_epochs=target,
            target_reached=completed >= target
            if isinstance(completed, int) and isinstance(target, int)
            else None,
            interpretation="Reaching the configured epoch target is separate from a recorded clean completion. No marker is invented for legacy runs.",
        )
        dims = spec.get("task_dims", {})
        regression = config.get("regression_tasks", spec.get("regression_tasks", []))
        meanings = {
            "family": "Pfam protein-family classification; number of output classes",
            "function": "EC enzyme-function classification; number of output classes",
            "stability": "Continuous stability regression; one predicted value"
            if "stability" in regression
            else "Stability classification; number of output classes",
        }
        if dims:
            report["prediction_tasks"] = [
                dict(
                    task=k,
                    output_dimensions=v,
                    meaning=meanings.get(k, "Task-specific output head"),
                )
                for k, v in dims.items()
            ]
        multi = config.get("multi_label_tasks")
        report["multilabel"] = dict(
            status="not_recorded"
            if multi is None
            else ("enabled" if multi else "disabled"),
            tasks=multi or [],
            meaning="Multi-label heads can assign several labels to one protein. An empty configured list means this run did not enable those heads; it is not missing telemetry.",
        )
    curve_paths = [
        p for p in [root / "scores/curves.csv", root / "curves.csv"] if p.exists()
    ]
    report["curves"] = curve_summary(curve_paths[0], warnings) if curve_paths else {}
    logs = sorted(
        set(root.glob("*.log"))
        | set(root.glob("log.txt"))
        | set((root / "logs").glob("*.log"))
    )
    report["logs"] = []
    for path in logs:
        try:
            item = log_summary(path)
            item["age_seconds"] = round(max(0, now - path.stat().st_mtime), 1)
            report["logs"].append(item)
        except OSError as exc:
            warnings.append(f"{path}: {exc}")
    evidence = logs + curve_paths + checkpoints
    if status_path.exists():
        evidence.append(status_path)
    age = min((max(0, now - p.stat().st_mtime) for p in evidence), default=None)
    report["activity"] = dict(
        latest_artifact_age_seconds=round(age, 1) if age is not None else None,
        quiet_threshold_minutes=quiet_minutes,
        quiet_threshold_exceeded=age is not None and age > quiet_minutes * 60,
        interpretation="Artifact writes are activity, not proof of optimizer progress. A single snapshot cannot establish liveness for legacy runs without locks.",
    )
    if lock == "held" and age is not None and age > quiet_minutes * 60:
        report["status"] = "active_quiet"
    if isinstance(session, dict) and lock in ("released", "absent"):
        if (
            not complete_path.exists()
            or status_path.stat().st_mtime >= complete_path.stat().st_mtime
        ):
            terminal = session.get("status")
            if terminal in ("failed", "interrupted", "wall_time"):
                report["status"] = terminal
            elif (
                terminal == "complete"
                and isinstance(completion, dict)
                and completion.get("status") == "complete"
            ):
                report["status"] = "complete"
    if report["status"] in ("unknown", "incomplete"):
        warnings.append(
            "No verified completion or active lock. Logs may describe a previous session; interruption versus failure is not established."
        )
    if not historical and not logs and not curve_paths:
        warnings.append("No training logs or curves found.")
    if any(log["error_evidence"] for log in report["logs"]):
        warnings.append(
            "Error evidence exists in log tails; it is not proof that the current session failed."
        )
    if any(log["numerical_evidence"] for log in report["logs"]):
        warnings.append(
            "Inspect recorded numerical-health log evidence; historical messages may predate a resume."
        )
    if historical:
        files = sorted(
            set((root / "scores").glob("*.json"))
            | set((root / "reports/benchmarks").glob("*/result.json"))
            | set((root / "reports/benchmarks").glob("*/execution.json"))
        )
        report["evaluations"] = [
            dict(
                path=str(p),
                data=read_json(p, warnings),
                provenance_status="Recorded evidence; not independently revalidated",
            )
            for p in files
        ]
        if not files:
            warnings.append("No recorded evaluation or benchmark JSON artifacts found.")
    return report
