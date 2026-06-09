#!/usr/bin/env python3
"""
protein_lm_data_preparer
------------------------

Reads the UniProt bacterial protein TSV and emits JSONL splits for the
protein language model and classifier.

Input:
    data/raw/uniprot_bacteria_50_512.tsv

Outputs (by default):
    data/processed/protein_lm/train.jsonl
    data/processed/protein_lm/val.jsonl
    data/processed/protein_lm/train_classified.jsonl
    data/processed/protein_lm/val_classified.jsonl

JSONL schema (per line):
    {
        "sequence": "<AA sequence>",
        "func_label": "ENZYME" | "NON_ENZYME",
        "topo_label": "TM" | "GLOBULAR",
        "entry": "<UniProt accession>",
        "organism": "<organism name>",
        "length": <int>
    }

Only the `sequence`, `func_label`, and `topo_label` fields are consumed by the
current ProteinDataset / ProteinClassificationDataset, but the extra metadata
is useful for inspection.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


TSV_PATH_DEFAULT = Path("data/raw/uniprot_bacteria_50_512.tsv")


@dataclass
class ProteinRecord:
    entry: str
    sequence: str
    organism: str
    length: int
    ec_number: str
    keywords: str
    subcellular_location: str


def _infer_func_label(ec_number: str) -> str:
    """
    Map UniProt EC number field to a coarse functional label.

    Any non-empty EC annotation is treated as ENZYME, otherwise NON_ENZYME.
    """
    ec_number = (ec_number or "").strip()
    return "ENZYME" if ec_number else "NON_ENZYME"


def _infer_topo_label(keywords: str, subcellular_location: str) -> str:
    """
    Map UniProt topology cues to a coarse topological label.

    If either keywords or subcellular location mention membrane / transmembrane
    terms, we assign TM; otherwise GLOBULAR.
    """
    text = f"{keywords or ''} {subcellular_location or ''}".lower()
    membrane_markers = [
        "membrane",
        "transmembrane",
        "cell inner membrane",
        "cell outer membrane",
        "single-pass membrane protein",
        "multi-pass membrane protein",
        "lipoprotein",
    ]
    if any(tok in text for tok in membrane_markers):
        return "TM"
    return "GLOBULAR"


def _read_uniprot_tsv(path: Path) -> List[ProteinRecord]:
    """
    Read the UniProt bacterial TSV file and return a list of ProteinRecord.

    Expected header columns (tab-separated):
        Entry, Reviewed, Entry Name, Protein names, Gene Names, Organism,
        Length, Sequence, EC number, Keywords,
        Subcellular location [CC], Features
    """
    if not path.exists():
        raise SystemExit(f"[error] TSV not found: {path}")

    records: List[ProteinRecord] = []
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        required = [
            "Entry",
            "Sequence",
            "Organism",
            "Length",
            "EC number",
            "Keywords",
            "Subcellular location [CC]",
        ]
        missing = [c for c in required if c not in reader.fieldnames]
        if missing:
            raise SystemExit(
                f"[error] TSV missing columns {missing}, found {reader.fieldnames}"
            )

        for row in reader:
            try:
                length = int(row.get("Length", "").strip() or "0")
            except ValueError:
                length = 0
            seq = (row.get("Sequence") or "").strip()
            if not seq:
                continue
            records.append(
                ProteinRecord(
                    entry=(row.get("Entry") or "").strip(),
                    sequence=seq,
                    organism=(row.get("Organism") or "").strip(),
                    length=length,
                    ec_number=(row.get("EC number") or "").strip(),
                    keywords=(row.get("Keywords") or "").strip(),
                    subcellular_location=(
                        row.get("Subcellular location [CC]") or ""
                    ).strip(),
                )
            )
    return records


def _split_records(
    records: List[ProteinRecord], val_frac: float, seed: int
) -> Tuple[List[ProteinRecord], List[ProteinRecord]]:
    """
    Randomly split records into train and validation sets.
    """
    if not 0.0 < val_frac < 1.0:
        raise ValueError(f"val_frac must be between 0 and 1, got {val_frac}")

    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    cut = int(len(indices) * (1.0 - val_frac))
    train_idx = set(indices[:cut])

    train_records: List[ProteinRecord] = []
    val_records: List[ProteinRecord] = []
    for i, rec in enumerate(records):
        (train_records if i in train_idx else val_records).append(rec)
    return train_records, val_records


def _record_to_json(rec: ProteinRecord) -> dict:
    """
    Convert a ProteinRecord into the JSONL schema expected by ProteinDataset.
    """
    func_label = _infer_func_label(rec.ec_number)
    topo_label = _infer_topo_label(rec.keywords, rec.subcellular_location)
    return {
        "sequence": rec.sequence,
        "func_label": func_label,
        "topo_label": topo_label,
        "entry": rec.entry,
        "organism": rec.organism,
        "length": rec.length,
    }


def _write_jsonl(path: Path, records: List[ProteinRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            obj = _record_to_json(rec)
            fh.write(json.dumps(obj) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Prepare JSONL datasets for protein_lm from UniProt TSV."
    )
    ap.add_argument(
        "--tsv", type=Path, default=TSV_PATH_DEFAULT, help="Input UniProt TSV path."
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed/protein_lm"),
        help="Output directory for JSONL files.",
    )
    ap.add_argument(
        "--val-frac",
        type=float,
        default=0.1,
        help="Fraction of data to use for validation.",
    )
    ap.add_argument("--seed", type=int, default=13, help="Random seed for shuffling.")
    args = ap.parse_args()

    records = _read_uniprot_tsv(args.tsv)
    if not records:
        raise SystemExit("[error] No records found in TSV.")

    train_records, val_records = _split_records(records, args.val_frac, args.seed)
    out_dir = args.out_dir

    # For the LM, we use the same schema as for the classifier;
    # the LM will simply ignore the labels if not needed.
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    train_cls_path = out_dir / "train_classified.jsonl"
    val_cls_path = out_dir / "val_classified.jsonl"

    _write_jsonl(train_path, train_records)
    _write_jsonl(val_path, val_records)
    _write_jsonl(train_cls_path, train_records)
    _write_jsonl(val_cls_path, val_records)

    print(
        f"[protein_lm_data_preparer] wrote {len(train_records)} train and {len(val_records)} val records"
    )
    print(f"  LM train:        {train_path}")
    print(f"  LM val:          {val_path}")
    print(f"  classifier train:{train_cls_path}")
    print(f"  classifier val:  {val_cls_path}")


if __name__ == "__main__":
    main()
