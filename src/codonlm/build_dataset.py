#!/usr/bin/env python3
"""
Group-aware split by genome to prevent leakage.
Creates train/val/test NPZ packs.

Args:
  --test_frac 0.1
  --val_frac 0.1
  --group_meta data/processed/cds_meta.tsv
"""

from pathlib import Path
import argparse
import csv
import numpy as np
import random

from src.codonlm.lossless_packing import (
    PACKING_METADATA_FIELDS,
    chunk_record,
    pack_chunks,
    packed_arrays,
    packing_metadata_rows,
)

def load_lines(path):
    """Loads whitespace-separated integers from each line of a file."""
    with open(path) as f:
        for line in f:
            yield [int(x) for x in line.strip().split()]

def main():
    """Builds and splits datasets by genome groups into train/val/test NPZ files."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="data/processed/codon_ids.txt")
    ap.add_argument("--group_meta", default="data/processed/cds_meta.tsv")
    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument("--windows_per_seq", type=int, default=2)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--test_frac", type=float, default=0.1)
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--pack_mode", choices=["multi", "single", "dynamic"], default="multi",
                    help="'multi': pack multiple CDS per window with <SEP>; 'single': one CDS per window; 'dynamic': save raw variable-length lists of arrays")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    # load sequences
    seqs = list(load_lines(args.ids))
    # load groups
    groups = []
    metadata_rows = []
    with open(args.group_meta, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"group_meta has no header: {args.group_meta}")
        group_col = "genome" if "genome" in reader.fieldnames else None
        if group_col is None and "genome_id" in reader.fieldnames:
            group_col = "genome_id"
        if group_col is None:
            raise ValueError(
                f"group_meta must contain a genome or genome_id column; found {reader.fieldnames}"
            )
        for row in reader:
            groups.append(row[group_col])
            metadata_rows.append(row)
    assert len(groups)==len(seqs), "meta and ids must align"

    # split by unique groups (fallback to sequence-level split if too few groups)
    uniq = list(sorted(set(groups)))
    rng.shuffle(uniq)

    buckets = {"train": [], "val": [], "test": []}

    def add_record(index: int, split: str) -> None:
        row = metadata_rows[index]
        tokens = seqs[index]
        codon_start = int(row.get("codon_start") or 0)
        codon_end = int(row.get("codon_end") or (codon_start + len(tokens) - 2))
        buckets[split].append(
            {
                "tokens": tokens,
                "source_id": row.get("source_id")
                or row.get("locus_tag")
                or row.get("record_id")
                or f"line:{index}",
                "source_line_idx": int(row.get("source_line_idx") or index),
                "fragment_line_idx": int(row.get("fragment_line_idx") or index),
                "fragment_index": int(row.get("fragment_index") or 0),
                "fragment_codon_start": codon_start,
                "fragment_codon_end": codon_end,
                "split": split,
            }
        )

    if len(uniq) < 3:
        # fallback to sequence-level split
        indices = list(range(len(seqs)))
        rng.shuffle(indices)
        n_test = max(1, int(len(seqs) * args.test_frac))
        n_val = max(1, int(len(seqs) * args.val_frac))

        test_idx = set(indices[:n_test])
        val_idx = set(indices[n_test:n_test + n_val])

        for i, _arr in enumerate(seqs):
            key = "val" if i in val_idx else "test" if i in test_idx else "train"
            add_record(i, key)
    else:
        n_test = max(1, int(len(uniq) * args.test_frac))
        n_val = max(1, int(len(uniq) * args.val_frac))
        # Ensure at least one group remains for training
        if n_test + n_val >= len(uniq):
            # reduce val first, then test if needed
            n_val = max(0, len(uniq) - 1 - n_test)
            if n_test + n_val >= len(uniq):
                n_test = max(0, len(uniq) - 1)

        val_groups = set(uniq[n_test:n_test + n_val])
        train_groups = set(uniq[n_test + n_val:])

        for index, g in enumerate(groups):
            key = "train" if g in train_groups else "val" if g in val_groups else "test"
            add_record(index, key)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.windows_per_seq != 1:
        print(
            f"[build] ignoring legacy windows_per_seq={args.windows_per_seq}; "
            "lossless packing emits every source transition once"
        )
    for name in ("train", "val", "test"):
        chunks = []
        for record in buckets[name]:
            chunks.extend(chunk_record(record, block_size=args.block_size))
        windows = pack_chunks(
            chunks, block_size=args.block_size, mode=args.pack_mode, sep_id=3
        )
        arrays = packed_arrays(
            windows, block_size=args.block_size, mode=args.pack_mode
        )
        out = out_dir / f"{name}_bs{args.block_size}.npz"
        np.savez_compressed(out, **arrays)
        metadata_path = out_dir / f"{name}_packing.tsv"
        with open(metadata_path, "w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=PACKING_METADATA_FIELDS, delimiter="\t"
            )
            writer.writeheader()
            writer.writerows(packing_metadata_rows(name, windows))
        print(
            f"[build] {name}: {len(chunks)} chunks in {len(windows)} windows "
            f"→ {out} (metadata {metadata_path})"
        )

if __name__ == "__main__":
    main()
