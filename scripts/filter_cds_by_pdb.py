#!/usr/bin/env python3
"""Prepare a CDS subset for PDB/structure-focused fine-tuning.

The current Stage 2.6 CDS metadata only contains ``line_idx`` and ``genome``.
That is enough to filter by explicit curated line indices, but not enough to
automatically join to UniProt/PDB metadata. This script supports the line-index
path now and validates metadata joins so future enriched metadata fails loudly
instead of producing a misleading empty subset.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_JOIN_KEYS = ("protein_id", "locus_tag", "gene", "gene_name", "entry")


@dataclass(frozen=True)
class CdsRecord:
    """One DNA row plus its aligned metadata."""

    source_line_idx: int
    dna: str
    meta: dict[str, str]


def read_dna_lines(path: Path) -> list[str]:
    """Read non-empty DNA rows."""
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a tab-separated metadata table."""
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")
        return list(reader.fieldnames), list(reader)


def read_line_indices(path: Path) -> set[int]:
    """Read one integer line index per row, allowing comments."""
    indices: set[int] = set()
    for line_no, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            indices.add(int(line))
        except ValueError as exc:
            raise ValueError(f"{path}:{line_no} is not an integer: {raw!r}") from exc
    return indices


def load_records(dna_path: Path, meta_path: Path) -> tuple[list[str], list[CdsRecord]]:
    """Load aligned DNA and metadata rows."""
    dna_lines = read_dna_lines(dna_path)
    header, meta_rows = read_tsv(meta_path)
    if len(dna_lines) != len(meta_rows):
        raise ValueError(
            f"DNA/meta row count mismatch: {len(dna_lines)} DNA rows vs "
            f"{len(meta_rows)} metadata rows"
        )

    records: list[CdsRecord] = []
    for i, (dna, meta) in enumerate(zip(dna_lines, meta_rows)):
        source_idx = int(meta.get("line_idx", i))
        records.append(CdsRecord(source_line_idx=source_idx, dna=dna, meta=meta))
    return header, records


def filter_by_line_indices(records: Iterable[CdsRecord], indices: set[int]) -> list[CdsRecord]:
    """Select records whose source line index is present in ``indices``."""
    selected = [record for record in records if record.source_line_idx in indices]
    missing = sorted(indices - {record.source_line_idx for record in selected})
    if missing:
        preview = ", ".join(str(x) for x in missing[:10])
        raise ValueError(f"{len(missing)} requested line indices were not found: {preview}")
    return selected


def validate_join_keys(meta_header: list[str], uniprot_header: list[str]) -> None:
    """Validate that metadata has a plausible key for UniProt/PDB matching."""
    meta_cols = {col.lower() for col in meta_header}
    uniprot_cols = {col.lower() for col in uniprot_header}
    usable_meta = sorted(meta_cols.intersection(DEFAULT_JOIN_KEYS))
    usable_uniprot = sorted(uniprot_cols.intersection({"entry", "gene names", "gene names (primary)"}))
    if not usable_meta:
        raise ValueError(
            "Cannot auto-filter by UniProt/PDB yet: CDS metadata has no protein/gene "
            f"join key. Found metadata columns: {', '.join(meta_header)}. "
            "Re-run extraction with protein_id, locus_tag, or gene annotations, or "
            "provide --structured_line_indices."
        )
    if not usable_uniprot:
        raise ValueError(
            "UniProt TSV does not expose a supported join key. Expected Entry or "
            f"Gene Names; found: {', '.join(uniprot_header)}."
        )


def write_subset(
    selected: list[CdsRecord],
    meta_header: list[str],
    out_dir: Path,
    manifest: dict[str, object],
) -> None:
    """Write DNA, metadata, and manifest outputs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dna_out = out_dir / "cds_dna.txt"
    meta_out = out_dir / "cds_meta.tsv"
    manifest_out = out_dir / "manifest.json"

    dna_out.write_text("".join(f"{record.dna}\n" for record in selected))

    retained_header = [col for col in meta_header if col != "line_idx"]
    out_header = ["line_idx", "source_line_idx", *retained_header]
    with meta_out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=out_header, delimiter="\t")
        writer.writeheader()
        for new_idx, record in enumerate(selected):
            row = {"line_idx": str(new_idx), "source_line_idx": str(record.source_line_idx)}
            row.update({col: record.meta.get(col, "") for col in retained_header})
            writer.writerow(row)

    manifest_out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dna", required=True, type=Path, help="Aligned CDS DNA text file")
    parser.add_argument("--meta", required=True, type=Path, help="Aligned CDS metadata TSV")
    parser.add_argument("--out_dir", required=True, type=Path, help="Output directory")
    parser.add_argument(
        "--structured_line_indices",
        type=Path,
        help="One source line_idx per row. This is the supported path for current metadata.",
    )
    parser.add_argument(
        "--uniprot_tsv",
        type=Path,
        help="Optional UniProt TSV for future metadata-enriched automatic filtering.",
    )
    return parser


def main() -> None:
    """Run the filter."""
    args = build_arg_parser().parse_args()
    meta_header, records = load_records(args.dna, args.meta)

    if args.structured_line_indices:
        indices = read_line_indices(args.structured_line_indices)
        selected = filter_by_line_indices(records, indices)
        mode = "line_indices"
    elif args.uniprot_tsv:
        uniprot_header, _ = read_tsv(args.uniprot_tsv)
        validate_join_keys(meta_header, uniprot_header)
        raise NotImplementedError(
            "UniProt/PDB matching is gated on enriched CDS metadata. Use "
            "--structured_line_indices for the current Stage 2.6 corpus."
        )
    else:
        raise SystemExit("Provide --structured_line_indices or --uniprot_tsv")

    manifest = {
        "mode": mode,
        "source_dna": str(args.dna),
        "source_meta": str(args.meta),
        "selected_records": len(selected),
        "total_records": len(records),
    }
    write_subset(selected, meta_header, args.out_dir, manifest)
    print(
        f"[filter] wrote {len(selected)}/{len(records)} CDS rows to {args.out_dir} "
        f"using {mode}"
    )


if __name__ == "__main__":
    main()
