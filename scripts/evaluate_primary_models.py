"""Evaluate named CodonLM checkpoints against frozen-token count baselines."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

from scripts.eval_ppl_baselines import evaluate_baselines, fit_baselines
from src.codonlm.dataset_manifest import load_dataset_manifest, manifest_artifact_path
from src.codonlm.evaluation_provenance import bind_dataset_manifest
from src.codonlm.training.vocabulary import resolve_vocabulary_contract
from src.training.inspection.benchmarks import run_benchmark


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must use NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise argparse.ArgumentTypeError(f"invalid checkpoint name: {name!r}")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"checkpoint does not exist: {path}")
    return name, path


def _markdown(report: dict) -> str:
    lines = [
        "# Corrected CodonLM primary evaluation",
        "",
        f"Status: **{report['status']}**",
        "",
        f"Manifest: `{report['dataset']['manifest']['path']}`",
        f"Dataset ID: `{report['dataset']['dataset_id']}`",
        f"Evaluation tokens: {report.get('evaluated_tokens', 'unavailable')}",
        "",
        "| Model | NLL (nats/codon) | PPL | Bits/codon | Δ NLL vs trigram |",
        "|---|---:|---:|---:|---:|",
    ]
    baseline_results = report.get("baselines", {}).get("results", {})
    rows = [
        (
            name,
            values.get("cross_entropy_nats"),
            values.get("perplexity"),
            values.get("bits_per_codon"),
        )
        for name, values in baseline_results.items()
    ]
    for name, item in report.get("models", {}).items():
        result = item.get("metrics", {})
        nll = result.get("test_nll")
        rows.append(
            (
                f"CodonLM {name}",
                nll,
                result.get("test_ppl"),
                nll / math.log(2) if isinstance(nll, (int, float)) else None,
            )
        )
    trigram_nll = baseline_results.get("Trigram", {}).get("cross_entropy_nats")
    for name, nll, ppl, bits in rows:
        delta = (
            nll - trigram_nll if nll is not None and trigram_nll is not None else None
        )
        fields = [nll, ppl, bits, delta]
        rendered = ["—" if value is None else f"{value:.6f}" for value in fields]
        lines.append(f"| {name} | " + " | ".join(rendered) + " |")
    lines.extend(["", "## Checkpoint evaluations", ""])
    for name, item in report.get("models", {}).items():
        lines.append(
            f"- `{name}`: {item['status']}; checkpoint SHA-256 `{item['checkpoint_sha256']}`"
        )
        if item.get("error"):
            lines.append(f"  - Error: {item['error']}")
    if report.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report["warnings"])
    return "\n".join(lines) + "\n"


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    os.replace(temporary, path)


def _git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def run_evaluation(
    *,
    manifest_path: Path,
    checkpoints: list[tuple[str, Path]],
    output_dir: Path,
    alpha: float = 0.01,
    timeout: float = 3600,
) -> dict:
    """Fit count baselines once and benchmark every selected model on one test set."""
    if not checkpoints:
        raise ValueError("at least one --checkpoint NAME=PATH is required")
    if len({name for name, _ in checkpoints}) != len(checkpoints):
        raise ValueError("checkpoint names must be unique")
    if alpha <= 0 or timeout <= 0:
        raise ValueError("alpha and timeout must be positive")

    manifest_path = manifest_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    manifest = load_dataset_manifest(manifest_path)
    train_path = manifest_artifact_path(
        manifest, manifest_path, "train_tokens"
    ).resolve()
    test_path = manifest_artifact_path(manifest, manifest_path, "test_tokens").resolve()
    vocab_path = manifest_artifact_path(manifest, manifest_path, "vocabulary").resolve()
    _, manifest_provenance = bind_dataset_manifest(
        manifest_path,
        expected_artifacts={"train_tokens": train_path, "test_tokens": test_path},
    )
    if output_dir.exists():
        raise ValueError(f"output directory must not already exist: {output_dir}")

    inputs = {
        "manifest": manifest_path,
        "train_tokens": train_path,
        "test_tokens": test_path,
        "vocabulary": vocab_path,
        **{f"checkpoint:{name}": path.resolve() for name, path in checkpoints},
    }
    input_hashes = {name: _sha256(path) for name, path in inputs.items()}
    contract = resolve_vocabulary_contract(
        [train_path, test_path], configured_path=vocab_path, configured_size=None
    )
    reset_token_ids = frozenset(
        index for index, token in enumerate(contract.tokens) if token == "<SEP>"
    )
    counts = fit_baselines(
        train_path,
        contract.size,
        alpha,
        reset_token_ids=reset_token_ids,
    )
    baseline_results, token_count, best_baseline = evaluate_baselines(
        test_path,
        counts,
        contract.size,
        alpha,
        reset_token_ids=reset_token_ids,
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "kind": "corrected_primary_model_evaluation",
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code_revision": _git_revision(),
        "python_version": platform.python_version(),
        "device": "cpu",
        "evaluator": str(Path(__file__).resolve()),
        "evaluator_sha256": _sha256(Path(__file__).resolve()),
        "dataset": {
            "dataset_id": manifest["dataset"]["id"],
            "manifest": {
                "path": str(manifest_path),
                "sha256": input_hashes["manifest"],
            },
            "provenance": manifest_provenance,
            "train_tokens": {
                "path": str(train_path),
                "sha256": input_hashes["train_tokens"],
            },
            "test_tokens": {
                "path": str(test_path),
                "sha256": input_hashes["test_tokens"],
            },
            "vocabulary": {
                "path": str(vocab_path),
                "sha256": input_hashes["vocabulary"],
            },
        },
        "baseline_protocol": {
            "estimator": "scripts.eval_ppl_baselines",
            "smoothing": {"method": "additive", "alpha": alpha},
            "context_boundary": {
                "method": "reset_history_after_tokens",
                "token_ids": sorted(reset_token_ids),
                "tokens": [contract.tokens[index] for index in sorted(reset_token_ids)],
            },
            "vocabulary": contract.provenance(),
        },
        "evaluated_tokens": token_count,
        "best_simple_baseline": best_baseline,
        "baselines": {
            "results": baseline_results,
            "train_path": str(train_path),
            "test_path": str(test_path),
        },
        "models": {},
        "warnings": [],
    }

    for name, checkpoint in checkpoints:
        model_root = output_dir / "model_runs" / name
        model_root.mkdir(parents=True, exist_ok=True)
        item = {
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": input_hashes[f"checkpoint:{name}"],
            "status": "running",
        }
        try:
            receipt = run_benchmark(
                model_root,
                "codonlm",
                "codon-test",
                checkpoint=checkpoint.resolve(),
                manifest=manifest_path,
                timeout=timeout,
            )
            item["benchmark_receipt"] = receipt
            item["status"] = receipt["status"]
            if receipt["status"] == "passed":
                metrics_path = Path(receipt["result"])
                metrics = json.loads(metrics_path.read_text())
                if not isinstance(metrics, dict):
                    raise ValueError("CodonLM evaluator returned non-mapping metrics")
                model_tokens = metrics.get("test_evaluated_tokens")
                if model_tokens != token_count:
                    raise ValueError(
                        f"evaluated token mismatch: baselines={token_count}, CodonLM={model_tokens}"
                    )
                item["metrics"] = metrics
                model_nll = metrics.get("test_nll")
                model_ppl = metrics.get("test_ppl")
                if not isinstance(model_nll, (int, float)) or not math.isfinite(
                    model_nll
                ):
                    raise ValueError("CodonLM evaluator returned invalid test_nll")
                if not isinstance(model_ppl, (int, float)) or not math.isfinite(
                    model_ppl
                ):
                    raise ValueError("CodonLM evaluator returned invalid test_ppl")
                trigram_nll = baseline_results["Trigram"]["cross_entropy_nats"]
                item["comparison"] = {
                    "bits_per_codon": model_nll / math.log(2),
                    "delta_nll_vs_trigram": model_nll - trigram_nll,
                    "beats_trigram": model_nll < trigram_nll,
                }
            elif receipt["status"] == "timeout":
                item["error"] = f"benchmark exceeded {timeout} seconds"
            else:
                item["error"] = "benchmark failed; see its receipt and log"
        except (OSError, ValueError, TypeError, KeyError) as exc:
            item.update(status="failed", error=str(exc))
        report["models"][name] = item

    try:
        report["input_hashes_unchanged"] = all(
            _sha256(path) == input_hashes[name] for name, path in inputs.items()
        )
    except OSError as exc:
        report["input_hashes_unchanged"] = False
        report["warnings"].append(f"Could not recheck all input hashes: {exc}")
    if not report["input_hashes_unchanged"]:
        report["warnings"].append("One or more inputs changed during evaluation.")
    model_success = all(
        item["status"] == "passed" for item in report["models"].values()
    )
    report["status"] = (
        "passed" if model_success and report["input_hashes_unchanged"] else "failed"
    )
    report["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report["output_dir"] = str(output_dir)
    _atomic_write(
        output_dir / "evaluation.json",
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    _atomic_write(output_dir / "evaluation.md", _markdown(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=_parse_checkpoint,
        required=True,
        metavar="NAME=PATH",
        help="Explicit checkpoint selection; repeat for each seed.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--timeout", type=float, default=3600)
    args = parser.parse_args()
    try:
        report = run_evaluation(
            manifest_path=args.manifest,
            checkpoints=args.checkpoint,
            output_dir=args.output_dir,
            alpha=args.alpha,
            timeout=args.timeout,
        )
    except (OSError, ValueError, TypeError) as exc:
        print(
            json.dumps({"status": "error", "error": str(exc)}, indent=2),
            file=sys.stderr,
        )
        return 2
    print(_markdown(report), end="")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
