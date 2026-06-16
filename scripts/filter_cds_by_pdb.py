#!/usr/bin/env python3
"""Prepare a CDS subset for PDB/structure-focused fine-tuning.

The preferred path is an exact translated-protein sequence match against a
local UniProt TSV containing a ``Sequence`` column and structure evidence
(``3D-structure`` keyword). Explicit curated line-index filters are still
supported for manual experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_JOIN_KEYS = ("protein_id", "locus_tag", "gene", "gene_name", "entry")

CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


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


def translate_dna(dna: str) -> str:
    """Translate DNA with the standard bacterial CDS convention."""
    dna = dna.strip().upper().replace("U", "T")
    aa: list[str] = []
    for i in range(0, (len(dna) // 3) * 3, 3):
        codon = dna[i : i + 3]
        residue = CODON_TABLE.get(codon)
        if residue is None:
            return ""
        if residue == "*":
            break
        aa.append(residue)
    return "".join(aa)


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
    has_sequence = "sequence" in uniprot_cols
    usable_uniprot = sorted(
        uniprot_cols.intersection({"entry", "gene names", "gene names (primary)", "sequence"})
    )
    if not usable_meta and not has_sequence:
        raise ValueError(
            "Cannot auto-filter by UniProt/PDB: CDS metadata has no protein/gene "
            "join key and UniProt TSV has no Sequence column. "
            f"Found metadata columns: {', '.join(meta_header)}."
        )
    if not usable_uniprot:
        raise ValueError(
            "UniProt TSV does not expose a supported join key. Expected Sequence, Entry, or "
            f"Gene Names; found: {', '.join(uniprot_header)}."
        )


def _has_structure_evidence(row: dict[str, str]) -> bool:
    text = " ".join(
        row.get(col, "")
        for col in ("Keywords", "Features", "Cross-reference (PDB)", "PDB")
    ).lower()
    return "3d-structure" in text or "pdb" in text


def filter_by_uniprot_sequence(
    records: Iterable[CdsRecord],
    uniprot_rows: Iterable[dict[str, str]],
) -> list[CdsRecord]:
    """Select CDS records whose translated AA exactly matches structured UniProt."""
    sequence_to_rows: dict[str, list[dict[str, str]]] = {}
    for row in uniprot_rows:
        seq = row.get("Sequence", "").strip().upper()
        if not seq or not _has_structure_evidence(row):
            continue
        sequence_to_rows.setdefault(seq, []).append(row)

    selected: list[CdsRecord] = []
    for record in records:
        aa = translate_dna(record.dna)
        if not aa:
            continue
        matches = sequence_to_rows.get(aa)
        if not matches:
            continue
        match = matches[0]
        meta = dict(record.meta)
        meta.update(
            {
                "aa_sequence": aa,
                "uniprot_entry": match.get("Entry", ""),
                "uniprot_entry_name": match.get("Entry Name", ""),
                "uniprot_protein_names": match.get("Protein names", ""),
                "uniprot_gene_names": match.get("Gene Names", ""),
                "uniprot_keywords": match.get("Keywords", ""),
            }
        )
        selected.append(CdsRecord(record.source_line_idx, record.dna, meta))
    return selected


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

    extra_header = [
        "aa_sequence",
        "uniprot_entry",
        "uniprot_entry_name",
        "uniprot_protein_names",
        "uniprot_gene_names",
        "uniprot_keywords",
    ]
    retained_header = [col for col in meta_header if col != "line_idx"]
    for col in extra_header:
        if any(col in record.meta for record in selected) and col not in retained_header:
            retained_header.append(col)
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
        uniprot_header, uniprot_rows = read_tsv(args.uniprot_tsv)
        validate_join_keys(meta_header, uniprot_header)
        selected = filter_by_uniprot_sequence(records, uniprot_rows)
        mode = "uniprot_sequence"
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
