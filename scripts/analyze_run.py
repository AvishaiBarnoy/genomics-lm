"""Inspect a historical run and optionally execute a selected CPU benchmark suite."""

import argparse
from pathlib import Path

from src.training.inspection.benchmarks import run_benchmark
from src.training.inspection.reporting import add_output_arguments, emit, error_report
from src.training.inspection.runs import inspect_run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--run-type", choices=["auto", "codonlm", "protein-critic"], default="auto"
    )
    parser.add_argument("--benchmark", choices=["codon-test", "critic-validation"])
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Explicit checkpoint path for benchmark execution",
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--benchmark-config", type=Path)
    parser.add_argument("--benchmark-data", type=Path)
    parser.add_argument("--benchmark-timeout", type=float, default=3600)
    add_output_arguments(parser)
    args = parser.parse_args()
    exit_code = 0
    try:
        report = inspect_run(args.run_dir, historical=True, run_type=args.run_type)
        if args.benchmark:
            if args.checkpoint is None:
                raise ValueError(
                    "--benchmark requires --checkpoint; checkpoint selection is explicit"
                )
            receipt = run_benchmark(
                Path(report["run_dir"]),
                report["run_type"],
                args.benchmark,
                checkpoint=args.checkpoint,
                manifest=args.manifest,
                config=args.benchmark_config,
                data=args.benchmark_data,
                timeout=args.benchmark_timeout,
            )
            report = inspect_run(args.run_dir, historical=True, run_type=args.run_type)
            report["benchmark_execution"] = receipt
            exit_code = 0 if receipt["status"] == "passed" else 1
    except (OSError, ValueError, TypeError) as exc:
        report = error_report("run_analysis", exc)
        exit_code = 2
    emit(report, args)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
