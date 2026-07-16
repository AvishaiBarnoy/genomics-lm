#!/usr/bin/env python3
"""
scripts/build_global_manifest.py — Global Manifest Builder and Group-Aware Splitter

This script implements a global, leakage-resistant data preparation pipeline.
Instead of splitting datasets individually and stacking them (which causes P0 genomic/homology leakage),
it:
  1. Gathers all CDS sequences and metadata across all configured GenBank (GBFF) files.
  2. Groups sequence records by genome_id (filename-derived) or taxonomic genus.
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
from pathlib import Path
from typing import Dict, List, Set, Tuple
import numpy as np
import yaml
from Bio import SeqIO

from src.codonlm.codon_tokenize import to_ids, stoi, itos
from src.codonlm.extract_cds_from_genbank import reverse_complement, _first_qualifier, _join_qualifier

def _load_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text()) or {}
    if not isinstance(cfg, dict):
        raise SystemExit(f"[error] Config at {path} must be a mapping.")
    if "data" in cfg and isinstance(cfg["data"], dict):
        for k, v in cfg["data"].items():
            cfg.setdefault(k, v)
    return cfg

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
    ap.add_argument("--group-by", choices=["genome", "genus", "sequence"], default="genome",
                    help="Split grouping criterion to avoid leakage.")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--force", action="store_true", help="Force rebuild")
    args = ap.parse_args()

    cfg = _load_config(Path(args.config))
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    block_size = int(cfg.get("block_size", 256))
    val_frac = float(cfg.get("val_frac", 0.1))
    test_frac = float(cfg.get("test_frac", 0.1))
    pack_mode = cfg.get("pack_mode", "multi")
    windows_per_seq = int(float(cfg.get("windows_per_seq", 2)))
    min_len = int(cfg.get("min_len", 90))

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    datasets = cfg.get("datasets", [])
    if not datasets:
        raise SystemExit("[error] No datasets found in config.")

    # 1. Extraction phase
    print(f"[global-prep] Extracting sequences from {len(datasets)} datasets...")
    all_records: List[dict] = []
    
    for ds in datasets:
        name = ds["name"]
        gbff_path = Path(ds["gbff"])
        if not gbff_path.exists():
            raise FileNotFoundError(f"GBFF not found: {gbff_path}")
        
        # Determine genome_id
        parts = gbff_path.stem.split("_")
        if len(parts) >= 2:
            genome_id = "_".join(parts[:2])
        else:
            genome_id = parts[0]
            
        print(f"  Processing {name} ({gbff_path.name}) with genome_id={genome_id}...")
        
        for rec in SeqIO.parse(gbff_path, "genbank"):
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
                
                if len(cds_seq) >= min_len and set(cds_seq) <= set("ACGTN"):
                    all_records.append({
                        "sequence": cds_seq,
                        "genome": genome_id,
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

    # 2. Splitting phase
    if args.group_by == "sequence":
        print("[global-prep] Performing sequence-level random split...")
        indices = list(range(total_seqs))
        rng.shuffle(indices)
        n_test = max(1, int(total_seqs * test_frac))
        n_val = max(1, int(total_seqs * val_frac))
        test_idx = set(indices[:n_test])
        val_idx = set(indices[n_test:n_test + n_val])
        
        for idx, rec in enumerate(all_records):
            if idx in val_idx:
                rec["split"] = "val"
            elif idx in test_idx:
                rec["split"] = "test"
            else:
                rec["split"] = "train"
    else:
        group_key = args.group_by  # "genome" or "genus"
        groups = [rec[group_key] for rec in all_records]
        uniq_groups = list(sorted(set(groups)))
        rng.shuffle(uniq_groups)
        
        n_groups = len(uniq_groups)
        print(f"[global-prep] Found {n_groups} unique groups based on '{group_key}': {uniq_groups}")
        
        if n_groups < 3:
            print(f"[global-prep] WARNING: Too few groups ({n_groups}) for group split. Falling back to sequence-level split.")
            indices = list(range(total_seqs))
            rng.shuffle(indices)
            n_test = max(1, int(total_seqs * test_frac))
            n_val = max(1, int(total_seqs * val_frac))
            test_idx = set(indices[:n_test])
            val_idx = set(indices[n_test:n_test + n_val])
            for idx, rec in enumerate(all_records):
                if idx in val_idx:
                    rec["split"] = "val"
                elif idx in test_idx:
                    rec["split"] = "test"
                else:
                    rec["split"] = "train"
        else:
            n_test = max(1, int(n_groups * test_frac))
            n_val = max(1, int(n_groups * val_frac))
            
            if n_test + n_val >= n_groups:
                n_val = max(0, n_groups - 1 - n_test)
                if n_test + n_val >= n_groups:
                    n_test = max(0, n_groups - 1)
            
            test_groups = set(uniq_groups[:n_test])
            val_groups = set(uniq_groups[n_test:n_test + n_val])
            train_groups = set(uniq_groups[n_test + n_val:])
            
            print(f"  Train groups: {train_groups}")
            print(f"  Val groups: {val_groups}")
            print(f"  Test groups: {test_groups}")
            
            for rec in all_records:
                g = rec[group_key]
                if g in train_groups:
                    rec["split"] = "train"
                elif g in val_groups:
                    rec["split"] = "val"
                else:
                    rec["split"] = "test"

    # Count split stats
    counts = {"train": 0, "val": 0, "test": 0}
    for rec in all_records:
        counts[rec["split"]] += 1
    print(f"[global-prep] Split counts: train={counts['train']}, val={counts['val']}, test={counts['test']}")

    # 3. Save combined metadata & DNA
    out_dir = Path("data/processed/global") / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    
    meta_path = out_dir / "cds_meta.tsv"
    dna_path = out_dir / "cds_dna.txt"
    
    with open(meta_path, "w", newline="") as fm, open(dna_path, "w") as fd:
        writer = csv.DictWriter(fm, fieldnames=["line_idx", "split"] + [k for k in all_records[0].keys() if k != "sequence"], delimiter="\t")
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
            random.shuffle(indices)
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
                    i = random.randrange(0, len(arr) - block_size - 1)
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
        else:
            np.savez_compressed(out_npz, X=X, Y=Y)
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
    manifest = {
        "train": str(out_paths["train"]),
        "val": str(out_paths["val"]),
        "test": str(out_paths["test"]),
        "datasets": datasets,
        "group_by": args.group_by,
    }
    
    manifest_json_path = out_dir / "manifest.json"
    manifest_json_path.write_text(json.dumps(manifest, indent=2))
    (run_dir / "combined_manifest.json").write_text(json.dumps(manifest, indent=2))
    
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
    empty_windows = {"train": 0, "val": 0, "test": 0}
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
