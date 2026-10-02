"""Explicit, isolated benchmark execution with persistent execution receipts."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import tempfile

from src.codonlm.dataset_manifest import load_dataset_manifest, manifest_artifact_path

from .reporting import observed_at
from .runs import lock_state, read_json

REPO_ROOT = Path(__file__).resolve().parents[3]


def fingerprint(path: Path) -> dict:
    """Hash an explicitly selected benchmark input without loading it into RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest()}


def run_benchmark(
    root: Path,
    run_type: str,
    suite: str,
    *,
    checkpoint: Path,
    manifest: Path | None = None,
    config: Path | None = None,
    data: Path | None = None,
    timeout: float = 3600,
) -> dict:
    """Run a supported CPU suite only when its inputs are explicit and available."""
    root = root.resolve()
    expected = {"codon-test": "codonlm", "critic-validation": "protein-critic"}
    if expected.get(suite) != run_type:
        raise ValueError(
            f"{suite} requires run type {expected.get(suite)}, found {run_type}"
        )
    if lock_state(root) == "held":
        raise ValueError("Refusing historical benchmarks while the run lock is held")
    if timeout <= 0:
        raise ValueError("--benchmark-timeout must be positive")
    if suite == "codon-test" and manifest is None:
        raise ValueError("codon-test requires --manifest")
    if suite == "critic-validation" and (config is None or data is None):
        raise ValueError(
            "critic-validation requires --benchmark-config and --benchmark-data"
        )
    inputs = {"checkpoint": checkpoint.resolve()}
    for name, value in [("manifest", manifest), ("config", config), ("data", data)]:
        if value is not None:
            inputs[name] = value.resolve()
    if suite == "codon-test":
        # Select the artifact here; the evaluator performs full manifest validation.
        selected_manifest = load_dataset_manifest(
            inputs["manifest"], verify_artifacts=False
        )
        inputs["test_npz"] = manifest_artifact_path(
            selected_manifest, inputs["manifest"], "test_tokens"
        ).resolve()
    for path in inputs.values():
        if not path.is_file():
            raise ValueError(f"Benchmark input does not exist: {path}")
    provenance = {name: fingerprint(path) for name, path in inputs.items()}
    parent = root / "reports/benchmarks"
    parent.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=suite + "-", dir=parent))
    result_path = directory / "result.json"
    if suite == "codon-test":
        (directory / "checkpoints").mkdir()
        (directory / "checkpoints/best.pt").symlink_to(inputs["checkpoint"])
        if (root / "itos.txt").exists():
            (directory / "itos.txt").symlink_to(root / "itos.txt")
        command = [
            sys.executable,
            "-m",
            "scripts.evaluate_test",
            "--run_dir",
            str(directory),
            "--manifest",
            str(inputs["manifest"]),
            "--test_npz",
            str(inputs["test_npz"]),
            "--checkpoint-name",
            "best.pt",
        ]
    else:
        command = [
            sys.executable,
            "-m",
            "scripts.eval_multi_task_critic",
            "--ckpt",
            str(inputs["checkpoint"]),
            "--config",
            str(inputs["config"]),
            "--val_data",
            str(inputs["data"]),
            "--split",
            "validation",
            "--device",
            "cpu",
            "--out_json",
            str(result_path),
        ]
    receipt = dict(
        suite=suite,
        status="running",
        started_at=observed_at(),
        command=command,
        cwd=str(REPO_ROOT),
        inputs=provenance,
        device="cpu",
        python_version=platform.python_version(),
        timeout_seconds=timeout,
        log=str(directory / "benchmark.log"),
        result=str(result_path),
        warnings=[],
    )
    receipt_path = directory / "execution.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    try:
        with (directory / "benchmark.log").open("w") as log:
            proc = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env={**os.environ, "FORCE_CPU": "1"},
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
        receipt["returncode"] = proc.returncode
        receipt["status"] = "passed" if proc.returncode == 0 else "failed"
        if suite == "codon-test" and proc.returncode == 0:
            source = directory / "scores/metrics.json"
            if source.exists():
                result_path.write_bytes(source.read_bytes())
        if receipt["status"] == "passed":
            result = read_json(result_path, receipt["warnings"])
            if not isinstance(result, dict) or not result:
                receipt["status"] = "failed"
                receipt["warnings"].append(
                    "Evaluator did not emit a nonempty result object"
                )
    except subprocess.TimeoutExpired:
        receipt["status"] = "timeout"
    except OSError as exc:
        receipt.update(status="failed", error=str(exc))
    finally:
        # A concurrently replaced checkpoint must not be attributed to the hash
        # captured before execution, even if the evaluator exited successfully.
        try:
            receipt["inputs_unchanged"] = all(
                fingerprint(path)["sha256"] == provenance[name]["sha256"]
                for name, path in inputs.items()
            )
        except OSError:
            receipt["inputs_unchanged"] = False
        if not receipt["inputs_unchanged"]:
            receipt["warnings"].append(
                "Benchmark inputs changed or disappeared during execution"
            )
            if receipt["status"] == "passed":
                receipt["status"] = "failed"
        receipt["finished_at"] = observed_at()
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
