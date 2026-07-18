#!/usr/bin/env python3
"""
scripts/build_global_manifest.py — Global Manifest Builder and Group-Aware Splitter

This script implements a global, leakage-resistant data preparation pipeline.
Instead of splitting datasets individually and stacking them (which causes P0 genomic/homology leakage),
it:
  1. Gathers all CDS sequences and metadata across all configured GenBank (GBFF) files.
  2. Resolves stable genome accessions and groups records by genome or genus.
  3. Splits groups globally into train/val/test partitions (Option A).
  4. Tokenizes all sequences using the standard codon tokenizer.
  5. Packs split tokenized IDs into final NPZ files directly.
  6. Emits pipeline_prepare.json and manifest.json for downstream training compatibility.

Usage:
  python -m scripts.build_global_manifest --config configs/tiny_mps.yaml --run-id my_run --run-dir runs/my_run --group-by genus
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import yaml
from Bio import SeqIO

from src.codonlm.codon_tokenize import to_ids, stoi, itos
from src.codonlm.extract_cds_from_genbank import reverse_complement, _first_qualifier, _join_qualifier


ASSEMBLY_ACCESSION_RE = re.compile(
    r"(?<![A-Z0-9])(GC[AF]_\d+(?:\.\d+)?)(?![\d.])", re.IGNORECASE
)

def _load_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text()) or {}
    if not isinstance(cfg, dict):
        raise SystemExit(f"[error] Config at {path} must be a mapping.")
    if "data" in cfg and isinstance(cfg["data"], dict):
        for k, v in cfg["data"].items():
            cfg.setdefault(k, v)
    return cfg


def _parse_extra_dataset(spec: str) -> dict:
    parts = spec.split(",")
    if len(parts) < 2:
        raise SystemExit(
            f"[error] Bad --extra-dataset spec (need name,gbff[,min_len]): {spec}"
        )
    entry: dict = {"name": parts[0], "gbff": parts[1]}
    if len(parts) > 2:
        entry["min_len"] = int(parts[2])
    return entry


def resolve_genome_identity(dataset: dict, gbff_path: Path, record) -> tuple[str, str]:
    """Resolve a stable genome identity and describe its provenance."""
    for key in ("genome_id", "assembly_accession", "accession"):
        value = str(dataset.get(key, "")).strip()
        if value:
            return value, f"config.{key}"

    for component in (gbff_path.name, *(parent.name for parent in gbff_path.parents)):
        match = ASSEMBLY_ACCESSION_RE.search(component)
        if match:
            return match.group(1).upper(), "path_accession"

    accessions = record.annotations.get("accessions", [])
    if isinstance(accessions, str):
        accessions = [accessions]
    for value in accessions:
        accession = str(value).strip()
        if accession:
            return accession, "genbank.annotations.accessions"

    record_id = str(getattr(record, "id", "")).strip()
    if record_id and record_id.lower() not in {"unknown", "<unknown id>"}:
        return record_id, "genbank.record_id"

    raise ValueError(
        f"Cannot resolve genome identity for {gbff_path}; set genome_id or assembly_accession in the dataset config"
    )


def _assign_sequence_splits(
    records: List[dict], rng: random.Random, val_frac: float, test_frac: float
) -> None:
    if len(records) < 3:
        raise ValueError("Sequence-level splitting requires at least 3 records")
    indices = list(range(len(records)))
    rng.shuffle(indices)
    n_test = min(max(1, int(len(records) * test_frac)), len(records) - 2)
    n_val = min(max(1, int(len(records) * val_frac)), len(records) - n_test - 1)
    test_idx = set(indices[:n_test])
    val_idx = set(indices[n_test : n_test + n_val])
    for idx, record in enumerate(records):
        record["split"] = "test" if idx in test_idx else "val" if idx in val_idx else "train"


def _assign_group_splits(
    records: List[dict],
    group_key: str,
    rng: random.Random,
    val_frac: float,
    test_frac: float,
) -> dict[str, set[str]]:
    groups = sorted({str(record[group_key]) for record in records})
    if len(groups) < 3:
        raise ValueError(
            f"Scientific splitting requires at least 3 distinct {group_key} groups; found {len(groups)}"
        )
    rng.shuffle(groups)
    n_test = min(max(1, int(len(groups) * test_frac)), len(groups) - 2)
    n_val = min(max(1, int(len(groups) * val_frac)), len(groups) - n_test - 1)
    split_groups = {
        "test": set(groups[:n_test]),
        "val": set(groups[n_test : n_test + n_val]),
        "train": set(groups[n_test + n_val :]),
    }
    for record in records:
        group = str(record[group_key])
        record["split"] = next(
            split for split, assigned in split_groups.items() if group in assigned
        )
    return split_groups

def extract_genus(rec) -> str:
    """Extract Genus from BioPython SeqRecord annotation taxonomy or organism."""
    organism = rec.annotations.get("organism", "")
    if organism:
        parts = organism.split()
        if parts:
            return parts[0]
    taxonomy = rec.annotations.get("taxonomy", [])
    if taxonomy:
        return taxonomy[0]
    return "Unknown"

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config file path")
    ap.add_argument("--run-id", required=True, help="Run identifier")
    ap.add_argument("--run-dir", required=True, help="Run output directory")
    ap.add_argument("--group-by", choices=["genome", "genus", "sequence"], default=None,
                    help="Split grouping criterion to avoid leakage.")
    ap.add_argument(
        "--allow-sequence-split",
        action="store_true",
        help="Explicitly allow a non-scientific sequence-level split when grouped splitting is impossible.",
    )
    ap.add_argument(
        "--extra-dataset", action="append", default=[], help="NAME,GBFF[,MIN_LEN]"
    )
    ap.add_argument(
        "--output-dir",
        help="Prepared dataset directory (default: data/processed/global/<run-id>).",
    )
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--force", action="store_true", help="Force rebuild")
    args = ap.parse_args()

    cfg = _load_config(Path(args.config))
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    block_size = int(cfg.get("block_size", 256))
    val_frac = float(cfg.get("val_frac", 0.1))
    test_frac = float(cfg.get("test_frac", 0.1))
    if not (0.0 < val_frac < 1.0 and 0.0 < test_frac < 1.0):
        raise SystemExit("[error] val_frac and test_frac must both be between 0 and 1")
    if val_frac + test_frac >= 1.0:
        raise SystemExit("[error] val_frac + test_frac must be less than 1")
    pack_mode = cfg.get("pack_mode", "multi")
    windows_per_seq = int(float(cfg.get("windows_per_seq", 2)))
    min_len = int(cfg.get("min_len", 90))
    requested_group_by = args.group_by or str(cfg.get("split_group_by", "genome"))
    if requested_group_by not in {"genome", "genus", "sequence"}:
        raise SystemExit(
            f"[error] split_group_by must be genome, genus, or sequence; got {requested_group_by!r}"
        )
    if requested_group_by == "sequence" and not args.allow_sequence_split:
        raise SystemExit(
            "[error] Sequence-level splitting is non-scientific and requires --allow-sequence-split"
        )

    rng = random.Random(args.seed)
    datasets = list(cfg.get("datasets", []))
    datasets.extend(_parse_extra_dataset(spec) for spec in args.extra_dataset)
    if not datasets:
        raise SystemExit("[error] No datasets found in config.")

    # 1. Extraction phase
    print(f"[global-prep] Extracting sequences from {len(datasets)} datasets...")
    all_records: List[dict] = []
    genome_sources: Dict[str, dict] = {}
    announced_genomes: set[str] = set()
    
    for ds in datasets:
        name = ds["name"]
        gbff_path = Path(ds["gbff"])
        if not gbff_path.exists():
            raise FileNotFoundError(f"GBFF not found: {gbff_path}")
        
        dataset_min_len = int(ds.get("min_len", min_len))
        for rec in SeqIO.parse(gbff_path, "genbank"):
            try:
                genome_id, identity_source = resolve_genome_identity(ds, gbff_path, rec)
            except ValueError as exc:
                raise SystemExit(f"[error] {exc}") from exc
            resolved_path = str(gbff_path.resolve())
            previous = genome_sources.get(genome_id)
            if previous is not None and previous["gbff"] != resolved_path:
                raise SystemExit(
                    f"[error] Genome identity collision for {genome_id!r}: "
                    f"{previous['gbff']} and {resolved_path}. Set distinct genome_id values explicitly."
                )
            genome_sources.setdefault(
                genome_id,
                {"gbff": resolved_path, "identity_source": identity_source},
            )
            if genome_id not in announced_genomes:
                print(
                    f"  Processing {name} ({gbff_path.name}) with "
                    f"genome_id={genome_id} ({identity_source})..."
                )
                announced_genomes.add(genome_id)
            seq = str(rec.seq).upper()
            genus = extract_genus(rec)
            
            for feat in rec.features:
                if feat.type != "CDS":
                    continue
                s, e = int(feat.location.start), int(feat.location.end)
                strand = int(feat.location.strand or 1)
                cds_seq = seq[s:e]
                if strand == -1:
                    cds_seq = reverse_complement(cds_seq)
                
                if len(cds_seq) >= dataset_min_len and set(cds_seq) <= set("ACGTN"):
                    all_records.append({
                        "sequence": cds_seq,
                        "genome": genome_id,
                        "genome_identity_source": identity_source,
                        "genus": genus,
                        "dataset": name,
                        "record_id": str(rec.id),
                        "protein_id": _first_qualifier(feat, "protein_id"),
                        "locus_tag": _first_qualifier(feat, "locus_tag"),
                        "gene": _first_qualifier(feat, "gene"),
                        "product": _first_qualifier(feat, "product"),
                        "translation": _first_qualifier(feat, "translation"),
                        "db_xref": _join_qualifier(feat, "db_xref"),
                        "start": s,
                        "end": e,
                        "strand": strand,
                    })

    total_seqs = len(all_records)
    print(f"[global-prep] Extracted {total_seqs} total CDS records.")
    if not all_records:
        raise SystemExit("[error] No eligible CDS records were extracted.")

    # 2. Splitting phase
    effective_group_by = requested_group_by
    split_groups: dict[str, set[str]] | None = None
    if requested_group_by == "sequence":
        print("[global-prep] Performing sequence-level random split...")
        _assign_sequence_splits(all_records, rng, val_frac, test_frac)
    else:
        try:
            split_groups = _assign_group_splits(
                all_records, requested_group_by, rng, val_frac, test_frac
            )
        except ValueError as exc:
            if not args.allow_sequence_split:
                raise SystemExit(f"[error] {exc}") from exc
            print(
                f"[global-prep] NON-SCIENTIFIC: falling back from {requested_group_by} "
                "to an explicitly allowed sequence-level split."
            )
            effective_group_by = "sequence"
            _assign_sequence_splits(all_records, rng, val_frac, test_frac)
        else:
            for split, groups in split_groups.items():
                print(f"  {split.title()} groups: {groups}")

    # Count split stats
    counts = {"train": 0, "val": 0, "test": 0}
    for rec in all_records:
        counts[rec["split"]] += 1
    print(f"[global-prep] Split counts: train={counts['train']}, val={counts['val']}, test={counts['test']}")

    # 3. Save combined metadata & DNA
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("data/processed/global") / args.run_id
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    
    meta_path = out_dir / "cds_meta.tsv"
    dna_path = out_dir / "cds_dna.txt"
    
    with open(meta_path, "w", newline="") as fm, open(dna_path, "w") as fd:
        metadata_fields = [
            key
            for key in all_records[0]
            if key not in {"sequence", "line_idx", "split"}
        ]
        writer = csv.DictWriter(
            fm,
            fieldnames=["line_idx", "split", *metadata_fields],
            delimiter="\t",
        )
        writer.writeheader()
        
        for idx, rec in enumerate(all_records):
            fd.write(rec["sequence"] + "\n")
            row = {k: v for k, v in rec.items() if k != "sequence"}
            row["line_idx"] = idx
            writer.writerow(row)
            
    print(f"[global-prep] Wrote global metadata to {meta_path} and DNA to {dna_path}")

    # 4. Tokenization phase
    print("[global-prep] Tokenizing sequences...")
    token_ids_list: List[List[int]] = []
    for rec in all_records:
        tids = to_ids(rec["sequence"])
        token_ids_list.append(tids)

    # Save global codon ids
    ids_path = out_dir / "codon_ids.txt"
    with open(ids_path, "w") as f:
        for tids in token_ids_list:
            f.write(" ".join(str(i) for i in tids) + "\n")

    # 5. Packing phase
    splits = {"train": [], "val": [], "test": []}
    for idx, rec in enumerate(all_records):
        splits[rec["split"]].append(token_ids_list[idx])

    def pack_multi(name: str, subset: List[List[int]]) -> Tuple[np.ndarray, np.ndarray]:
        SEP_ID = 3
        PAD_ID = 0
        seqs = [arr for arr in subset if len(arr) > 2]
        if not seqs:
            return np.zeros((0, block_size), dtype=np.int32), np.zeros((0, block_size), dtype=np.int32)
            
        windows_goal = max(1, windows_per_seq * len(seqs))
        Xs, Ys = [], []
        indices = list(range(len(seqs)))
        offsets = [0] * len(seqs)
        
        for _ in range(windows_goal):
            rng.shuffle(indices)
            buf: List[int] = []
            for idx in indices:
                if len(buf) >= block_size:
                    break
                arr = seqs[idx]
                off = offsets[idx]
                if off >= len(arr):
                    continue
                room = block_size - len(buf)
                take = min(room, len(arr) - off)
                if take <= 0:
                    continue
                buf.extend(arr[off : off + take])
                offsets[idx] += take
                if offsets[idx] >= len(arr) and len(buf) < block_size:
                    buf.append(SEP_ID)
            if len(buf) < 2:
                break
            x = buf[:-1]
            y = buf[1:]
            if len(x) < block_size:
                pad_n = block_size - len(x)
                x = x + [PAD_ID] * pad_n
                y = y + [PAD_ID] * pad_n
            else:
                x = x[:block_size]
                y = y[:block_size]
            Xs.append(x)
            Ys.append(y)
            
            remaining = sum(max(0, len(arr) - offsets[i]) for i, arr in enumerate(seqs))
            if remaining < 2:
                break
                
        return np.array(Xs, dtype=np.int32), np.array(Ys, dtype=np.int32)

    def pack_single(name: str, subset: List[List[int]]) -> Tuple[np.ndarray, np.ndarray]:
        Xs, Ys = [], []
        for arr in subset:
            if len(arr) <= 2:
                continue
            for _ in range(windows_per_seq):
                if len(arr) <= block_size + 1:
                    x = arr[:-1]
                    y = arr[1:]
                    pad = [0] * max(0, block_size - len(x))
                    x = (x + pad)[:block_size]
                    y = (y + pad)[:block_size]
                else:
                    i = rng.randrange(0, len(arr) - block_size - 1)
                    x = arr[i : i + block_size]
                    y = arr[i + 1 : i + 1 + block_size]
                Xs.append(x)
                Ys.append(y)
        return np.array(Xs, dtype=np.int32), np.array(Ys, dtype=np.int32)

    def pack_dynamic(name: str, subset: List[List[int]]) -> Tuple[np.ndarray, np.ndarray]:
        filtered = []
        for arr in subset:
            if len(arr) <= 2:
                continue
            if len(arr) > block_size:
                arr = arr[-block_size:]
            filtered.append(arr)
        if not filtered:
            return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32)
        flat_X = np.concatenate([np.array(x, dtype=np.int32) for x in filtered])
        lengths = np.array([len(x) for x in filtered], dtype=np.int32)
        return flat_X, lengths

    out_paths = {}
    empty_windows: dict[str, int] = {}
    for name in ("train", "val", "test"):
        if pack_mode == "single":
            X, Y = pack_single(name, splits[name])
        elif pack_mode == "dynamic":
            X, Y = pack_dynamic(name, splits[name])  # Y holds lengths
        else:
            X, Y = pack_multi(name, splits[name])
            
        out_npz = out_dir / f"{name}_bs{block_size}.npz"
        if pack_mode == "dynamic":
            np.savez_compressed(out_npz, X=X, lengths=Y)
            empty_windows[name] = int(np.count_nonzero(Y == 0))
        else:
            np.savez_compressed(out_npz, X=X, Y=Y)
            empty_windows[name] = int(np.count_nonzero((Y != 0).sum(axis=1) == 0))
        out_paths[name] = out_npz
        print(f"[global-prep] Packed split {name} to {out_npz} with shape {X.shape}")

    # Write itos/vocab files
    itos_path = out_dir / "itos.txt"
    vocab_path = out_dir / "vocab_codon.txt"
    
    with open(itos_path, "w") as f:
        for i in sorted(itos.keys()):
            f.write(itos[i] + "\n")
            
    with open(vocab_path, "w") as f:
        for k, v in stoi.items():
            f.write(f"{k} {v}\n")

    # 6. Save combined manifest & pipeline_prepare.json
    achieved_fractions = {
        split: (counts[split] / total_seqs if total_seqs else 0.0)
        for split in ("train", "val", "test")
    }
    group_counts = None
    groups_by_split = None
    if split_groups is not None and effective_group_by != "sequence":
        groups_by_split = {
            split: sorted(groups) for split, groups in split_groups.items()
        }
        group_counts = {
            split: len(groups) for split, groups in split_groups.items()
        }

    manifest = {
        "train": str(out_paths["train"]),
        "val": str(out_paths["val"]),
        "test": str(out_paths["test"]),
        "datasets": datasets,
        "seed": args.seed,
        "split_policy": {
            "requested_group_by": requested_group_by,
            "effective_group_by": effective_group_by,
            "allow_sequence_split": bool(args.allow_sequence_split),
            "scientific_valid": effective_group_by != "sequence",
            "requested_fractions": {"val": val_frac, "test": test_frac},
            "achieved_record_fractions": achieved_fractions,
            "record_counts": counts,
            "group_counts": group_counts,
            "groups_by_split": groups_by_split,
        },
        "genome_sources": genome_sources,
        "packing": {
            "mode": pack_mode,
            "block_size": block_size,
            "windows_per_seq": windows_per_seq,
            "seed": args.seed,
        },
    }
    
    manifest_json_path = out_dir / "manifest.json"
    manifest_json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    (run_dir / "combined_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )
    
    pipeline_prepare_json = {
        "train_npz": str(out_paths["train"]),
        "val_npz": str(out_paths["val"]),
        "test_npz": str(out_paths["test"]),
        "primary_dna": str(dna_path),
        "combined_manifest": str(manifest_json_path),
    }
    
    result_path = run_dir / "pipeline_prepare.json"
    result_path.write_text(json.dumps(pipeline_prepare_json, indent=2))
    
    # Save integrity check data
    integrity = {
        "train_npz": str(out_paths["train"]),
        "val_npz": str(out_paths["val"]),
        "test_npz": str(out_paths["test"]),
        "empty_windows": empty_windows,
    }
    (run_dir / "integrity.json").write_text(json.dumps(integrity, indent=2))
    
    print(f"[global-prep] Completed global manifest data preparation. Output summary in {result_path}")

if __name__ == "__main__":
    main()
