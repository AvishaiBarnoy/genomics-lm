"""Inspect the current PR revision through GitHub CLI; never merge or modify it."""

from __future__ import annotations

import argparse
import json
import subprocess

from src.training.inspection.reporting import (
    add_output_arguments,
    emit,
    error_report,
    observed_at,
)


def summarize_ci(pr: dict, expected: list[str]) -> dict:
    """Classify a single PR rollup snapshot, retaining non-passing check states."""
    checks = []
    for item in pr.get("statusCheckRollup") or []:
        name = item.get("name") or item.get("context") or "<unnamed>"
        raw = (
            item.get("conclusion")
            if item.get("status") == "COMPLETED"
            else item.get("status")
        )
        raw = raw or item.get("state") or "UNKNOWN"
        status = {
            "SUCCESS": "passed",
            "FAILURE": "failed",
            "ERROR": "failed",
            "TIMED_OUT": "failed",
            "ACTION_REQUIRED": "failed",
            "STARTUP_FAILURE": "failed",
            "CANCELLED": "cancelled",
            "SKIPPED": "skipped",
            "NEUTRAL": "neutral",
            "QUEUED": "pending",
            "IN_PROGRESS": "pending",
            "PENDING": "pending",
            "WAITING": "pending",
            "REQUESTED": "pending",
        }.get(raw, "unknown")
        checks.append(
            dict(
                name=name,
                status=status,
                raw_state=raw,
                url=item.get("detailsUrl") or item.get("targetUrl"),
                started_at=item.get("startedAt"),
                completed_at=item.get("completedAt"),
            )
        )
    checks.sort(key=lambda c: (c["name"], c["started_at"] or "", c["url"] or ""))
    missing = sorted(set(expected) - {c["name"] for c in checks})
    states = {c["status"] for c in checks}
    status = "passed"
    for candidate in [
        "failed",
        "cancelled",
        "pending",
        "unknown",
        "skipped",
        "neutral",
    ]:
        if candidate in states:
            status = candidate
            break
    if status == "passed" and (missing or not checks):
        status = "missing"
    return dict(
        schema_version=1,
        kind="ci_check",
        observed_at=observed_at(),
        status=status,
        pr={
            k: pr.get(k)
            for k in [
                "number",
                "url",
                "title",
                "headRefOid",
                "state",
                "isDraft",
                "reviewDecision",
                "mergeable",
                "mergeStateStatus",
            ]
        },
        expected_checks=sorted(set(expected)),
        missing_checks=missing,
        checks=checks,
        scope="Current PR status rollup. Expected check names are not branch-protection rules; "
        "CI success does not imply merge approval. Core CI excludes slow/external/MPS tests.",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pr", help="PR number, URL, or branch; defaults to the current branch"
    )
    parser.add_argument("--repo", help="GitHub OWNER/REPO (optional host prefix)")
    parser.add_argument(
        "--expect",
        action="append",
        help="Expected check name; repeat to override lint/core-tests",
    )
    parser.add_argument("--timeout", type=float, default=60)
    add_output_arguments(parser)
    args = parser.parse_args()
    try:
        if args.timeout <= 0:
            raise ValueError("--timeout must be positive")
        command = ["gh", "pr", "view"]
        if args.pr:
            command.append(args.pr)
        if args.repo:
            command.extend(["--repo", args.repo])
        command.extend(
            [
                "--json",
                "number,url,title,headRefOid,state,isDraft,reviewDecision,mergeable,mergeStateStatus,statusCheckRollup",
            ]
        )
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=args.timeout, check=False
        )
        if result.returncode:
            raise RuntimeError(
                result.stderr.strip() or f"gh exited {result.returncode}"
            )
        report = summarize_ci(
            json.loads(result.stdout), args.expect or ["lint", "core-tests"]
        )
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        report = error_report("ci_check", exc)
    emit(report, args)
    return (
        0 if report["status"] == "passed" else (2 if report["status"] == "error" else 1)
    )


if __name__ == "__main__":
    raise SystemExit(main())
