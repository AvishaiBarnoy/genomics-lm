"""Read-only live-training check based on logs, curves, checkpoint ages and locks."""

import argparse
from pathlib import Path

from src.training.inspection.reporting import add_output_arguments, emit, error_report
from src.training.inspection.runs import inspect_run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--run-type", choices=["auto", "codonlm", "protein-critic"], default="auto"
    )
    parser.add_argument("--quiet-minutes", type=float, default=30)
    add_output_arguments(parser)
    args = parser.parse_args()
    try:
        report = inspect_run(
            args.run_dir, run_type=args.run_type, quiet_minutes=args.quiet_minutes
        )
    except (OSError, ValueError, TypeError) as exc:
        report = error_report("training_check", exc)
    emit(report, args)
    return 2 if report["status"] == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
