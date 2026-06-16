#!/usr/bin/env python3
"""Prepare structural-aware ProteinCritic labels from a UniProt TSV."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


PROTEIN_TYPE_LABELS = [
    "structured_pdb",
    "membrane",
    "signal_secreted",
    "disordered_low_complexity",
    "enzyme",
    "short_peptide",
    "soluble_candidate",
]


def _text(row: dict[str, str], *columns: str) -> str:
    return " ".join(row.get(column, "") for column in columns).lower()


def protein_type_labels(row: dict[str, str], short_len: int = 50) -> dict[str, int]:
    """Derive coarse structural protein-type labels from UniProt columns."""
    keywords = _text(row, "Keywords")
    features = _text(row, "Features")
    location = _text(row, "Subcellular location [CC]")
    names = _text(row, "Protein names")
    all_text = " ".join([keywords, features, location, names])

    length = int(row.get("Length") or len(row.get("Sequence", "")) or 0)
    labels = {
        "structured_pdb": int("3d-structure" in all_text or "pdb" in all_text),
        "membrane": int("membrane" in all_text or "transmembrane" in all_text),
        "signal_secreted": int(
            "signal" in features
            or "secreted" in location
            or "periplasm" in location
        ),
        "disordered_low_complexity": int(
            "low complexity" in features
            or "repeat" in features
            or "coiled coil" in features
            or "disorder" in all_text
            or "intrinsically disordered" in all_text
        ),
        "enzyme": int(bool(row.get("EC number", "").strip())),
        "short_peptide": int(length > 0 and length < short_len),
    }
    labels["soluble_candidate"] = int(
        length >= short_len
        and not labels["membrane"]
        and not labels["signal_secreted"]
        and not labels["disordered_low_complexity"]
    )
    return labels


def row_to_sample(row: dict[str, str], short_len: int) -> dict[str, object] | None:
    """Convert one UniProt row into a ProteinCritic JSONL sample."""
    sequence = row.get("Sequence", "").strip().upper()
    if not sequence:
        return None
    labels = protein_type_labels(row, short_len=short_len)
    return {
        "sequence": sequence,
        "entry": row.get("Entry", ""),
        "entry_name": row.get("Entry Name", ""),
        "protein_type": [labels[name] for name in PROTEIN_TYPE_LABELS],
        "protein_type_labels": labels,
    }


def read_uniprot_tsv(path: Path) -> list[dict[str, str]]:
    """Read a UniProt TSV export."""
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_jsonl(path: Path, samples: list[dict[str, object]]) -> None:
    """Write samples to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uniprot_tsv", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--short_len", type=int, default=50)
    args = parser.parse_args()

    samples = [
        sample
        for row in read_uniprot_tsv(args.uniprot_tsv)
        if (sample := row_to_sample(row, short_len=args.short_len)) is not None
    ]
    rng = random.Random(args.seed)
    rng.shuffle(samples)

    n_val = max(1, int(len(samples) * args.val_frac)) if len(samples) > 1 else 0
    val = samples[:n_val]
    train = samples[n_val:]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "train.jsonl", train)
    write_jsonl(args.out_dir / "val.jsonl", val)
    (args.out_dir / "task_vocabs.json").write_text(
        json.dumps(
            {
                "pfam": {"unknown": 0},
                "ec": {"unknown": 0},
                "stability": {"unknown": 0},
                "protein_type": {name: i for i, name in enumerate(PROTEIN_TYPE_LABELS)},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(
        f"[protein-type] wrote {len(train)} train and {len(val)} val samples to {args.out_dir}"
    )


if __name__ == "__main__":
    main()
