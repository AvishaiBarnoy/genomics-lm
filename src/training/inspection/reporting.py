"""Shared CLI and report contract for repository check-ups."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path


def observed_at() -> str:
    """Return an explicit UTC observation timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clean(value):
    """Keep output strict JSON, including metrics containing NaN/Infinity."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def _cell(value) -> str:
    if value is None:
        return "—"
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("|", "\\|")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
    )


def duration(seconds: float) -> str:
    """Display elapsed time in units people can compare."""
    remaining = max(0, int(seconds))
    parts = []
    for label, unit in [("d", 86400), ("h", 3600), ("m", 60), ("s", 1)]:
        count, remaining = divmod(remaining, unit)
        if count or (label == "s" and not parts):
            parts.append(f"{count}{label}")
        if len(parts) == 2:
            break
    return " ".join(parts)


def _display(key, value):
    if key.endswith("age_seconds") and isinstance(value, (int, float)):
        return duration(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    return value


def render_markdown(report: dict) -> str:
    """Render readable tables and lists from the same complete JSON evidence."""
    lines = [
        f"# {report.get('kind', 'Check-up').replace('_', ' ').title()}",
        "",
        f"**Status:** {_cell(report['status'])}",
        "",
    ]

    def section(value, level=2):
        if isinstance(value, dict):
            if not value:
                lines.extend(["Not recorded in the available artifacts.", ""])
            scalars = {
                k: v for k, v in value.items() if not isinstance(v, (dict, list))
            }
            if scalars:
                lines.extend(["| Field | Value |", "| --- | --- |"])
                lines.extend(
                    f"| {_cell(k.replace('_seconds', '') if k.endswith('age_seconds') else k)} | {_cell(_display(k, v))} |"
                    for k, v in scalars.items()
                )
                lines.append("")
            for key, nested in value.items():
                if isinstance(nested, (dict, list)):
                    lines.extend(
                        [f"{'#' * min(level, 6)} {key.replace('_', ' ').title()}", ""]
                    )
                    section(nested, level + 1)
        elif isinstance(value, list):
            if not value:
                lines.extend(["None recorded.", ""])
            elif all(
                isinstance(v, dict)
                and all(not isinstance(x, (dict, list)) for x in v.values())
                for v in value
            ):
                keys = list(dict.fromkeys(k for item in value for k in item))
                lines.extend(
                    [
                        "| "
                        + " | ".join(
                            _cell(
                                k.replace("_seconds", "")
                                if k.endswith("age_seconds")
                                else k
                            )
                            for k in keys
                        )
                        + " |",
                        "| " + " | ".join("---" for _ in keys) + " |",
                    ]
                )
                lines.extend(
                    "| "
                    + " | ".join(_cell(_display(k, item.get(k))) for k in keys)
                    + " |"
                    for item in value
                )
                lines.append("")
            else:
                for index, item in enumerate(value):
                    if isinstance(item, (dict, list)):
                        lines.extend([f"{'#' * min(level, 6)} Item {index + 1}", ""])
                        section(item, level + 1)
                    else:
                        lines.append(f"- {_cell(item)}")
                lines.append("")
        else:
            lines.extend([_cell(value), ""])

    section({k: v for k, v in clean(report).items() if k not in ("kind", "status")})
    return "\n".join(lines)


def add_output_arguments(parser) -> None:
    """Install the common rendering flags."""
    parser.add_argument("--format", choices=["md", "json"], default="md")
    parser.add_argument(
        "--output", type=Path, help="Save the report instead of printing it"
    )


def emit(report: dict, args) -> None:
    """Render a single report to stdout or an explicitly requested file."""
    report = clean(report)
    content = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        if args.format == "json"
        else render_markdown(report)
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)
    else:
        print(content, end="")


def error_report(kind: str, exc: Exception) -> dict:
    """Represent collection errors without pretending the check passed."""
    return dict(
        schema_version=1,
        kind=kind,
        observed_at=observed_at(),
        status="error",
        error=str(exc),
    )
