#!/usr/bin/env python3
"""
Prepare datasets for the training pipeline.

Responsibilities:
  * read YAML config + optional CLI dataset overrides
  * extract CDS + metadata (if missing or force)
  * tokenize into codon IDs / vocabulary
  * build train/val/test NPZ windows per dataset
  * concatenate NPZs across datasets into a combined manifest scoped to the run id
  * emit a small JSON summary in runs/<RUN_ID>/pipeline_prepare.json

The shell wrapper calls this module instead of embedding large Python blocks.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml


def _load_config(path: Path) -> Dict:
    cfg = yaml.safe_load(path.read_text()) or {}
    if not isinstance(cfg, dict):
        raise SystemExit(f"[error] Config at {path} must be a mapping.")
    return cfg


def _parse_extra_dataset(spec: str) -> Dict[str, object]:
    parts = spec.split(",")
    if len(parts) < 2:
        raise SystemExit(f"[error] Bad --extra-dataset spec (need name,gbff[,min_len]): {spec}")
    name, gbff = parts[0], parts[1]
    entry: Dict[str, object] = {"name": name, "gbff": gbff}
    if len(parts) > 2:
        entry["min_len"] = int(parts[2])
    return entry


def _make_dataset_entry(entry: Dict[str, object], block_size: int) -> Dict[str, str]:
    name = entry["name"]
    gbff = entry["gbff"]
    min_len = int(entry.get("min_len", 90))
    out_dir = Path("data/processed") / name
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "name": name,
        "gbff": str(Path(gbff)),
        "min_len": min_len,
        "out_dir": str(out_dir),
        "dna": str(out_dir / "cds_dna.txt"),
        "meta": str(out_dir / "cds_meta.tsv"),
        "ids": str(out_dir / "codon_ids.txt"),
        "vocab": str(out_dir / "vocab_codon.txt"),
        "itos": str(out_dir / "itos.txt"),
        "train": str(out_dir / f"train_bs{block_size}.npz"),
        "val": str(out_dir / f"val_bs{block_size}.npz"),
        "test": str(out_dir / f"test_bs{block_size}.npz"),
    }


def _ensure_dataset(entry: Dict[str, str], windows_per_seq: int, val_frac: float, test_frac: float, block_size: int, force: bool) -> None:
    """Run extraction/tokenization/build for a dataset."""
    dna = Path(entry["dna"])
    meta = Path(entry["meta"])
    ids = Path(entry["ids"])
    vocab = Path(entry["vocab"])
    itos = Path(entry["itos"])
    train = Path(entry["train"])
    val = Path(entry["val"])
    test = Path(entry["test"])

    cmds: List[List[str]] = []
    if force or not (dna.exists() and meta.exists()):
        cmds.append([
            "python", "-m", "src.codonlm.extract_cds_from_genbank",
            "--gbff", entry["gbff"],
            "--out_txt", str(dna),
            "--out_meta", str(meta),
            "--min_len", str(entry["min_len"]),
        ])
    else:
        print(f"[prepare] skip extract {entry['name']}")

    if force or not ids.exists():
        cmds.append([
            "python", "-m", "src.codonlm.codon_tokenize",
            "--inp", str(dna),
            "--out_ids", str(ids),
            "--out_vocab", str(vocab),
            "--out_itos", str(itos),
        ])
    else:
        print(f"[prepare] skip tokenize {entry['name']}")

    if force or not (train.exists() and val.exists() and test.exists()):
        cmds.append([
            "python", "-m", "src.codonlm.build_dataset",
            "--ids", str(ids),
            "--group_meta", str(meta),
            "--block_size", str(block_size),
            "--windows_per_seq", str(windows_per_seq),
            "--val_frac", str(val_frac),
            "--test_frac", str(test_frac),
            "--out_dir", str(train.parent),
        ])
    else:
        print(f"[prepare] skip build {entry['name']}")

    for cmd in cmds:
        print(f"[prepare] run {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


def _stack_npz(paths: List[str], out_path: Path) -> None:
    if not paths:
        raise ValueError("No NPZ paths provided for stacking")

    total = 0
    x_tail = None
    y_tail = None
    x_dtype = None
    y_dtype = None

    # First pass: determine total rows and shapes without keeping arrays resident
    for path in paths:
        with np.load(path, allow_pickle=False) as blob:
            X = np.asarray(blob["X"])
            Y = np.asarray(blob["Y"])
            if x_tail is None:
                x_tail = X.shape[1:]
                y_tail = Y.shape[1:]
                x_dtype = X.dtype
                y_dtype = Y.dtype
            total += X.shape[0]

    if total == 0:
        np.savez_compressed(out_path, X=np.zeros((0,) + x_tail, dtype=x_dtype), Y=np.zeros((0,) + y_tail, dtype=y_dtype))
        return

    X_out = np.empty((total,) + x_tail, dtype=x_dtype)
    Y_out = np.empty((total,) + y_tail, dtype=y_dtype)

    # Second pass: copy chunk-by-chunk to limit peak memory usage
    offset = 0
    for path in paths:
        with np.load(path, allow_pickle=False) as blob:
            X = np.asarray(blob["X"])
            Y = np.asarray(blob["Y"])
            rows = X.shape[0]
            X_out[offset:offset + rows] = X
            Y_out[offset:offset + rows] = Y
            offset += rows

    np.savez_compressed(out_path, X=X_out, Y=Y_out)


def _write_manifest(run_dir: Path, datasets: List[Dict[str, str]], block_size: int, windows_per_seq: int, val_frac: float, test_frac: float, force: bool) -> Path:
    manifest = {
        "datasets": datasets,
        "block_size": block_size,
        "windows_per_seq": windows_per_seq,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "force": int(force),
    }
    target = run_dir / "datasets_manifest.json"
    target.write_text(json.dumps(manifest, indent=2))
    return target


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare datasets for main.sh (data prep phase)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--extra-dataset", action="append", default=[], help="NAME,GBFF[,MIN_LEN]")
    args = ap.parse_args()

    config_path = Path(args.config)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = _load_config(config_path)

    block_size = int(cfg.get("block_size", 256))
    windows_raw = cfg.get("windows_per_seq", 2)
    try:
        windows_per_seq = int(float(windows_raw))
    except (TypeError, ValueError):
        raise SystemExit(f"[error] windows_per_seq must be numeric, got {windows_raw!r}")
    if windows_per_seq <= 0:
        raise SystemExit(f"[error] windows_per_seq must be positive, got {windows_per_seq}")
    val_frac = float(cfg.get("val_frac", 0.1))
    test_frac = float(cfg.get("test_frac", 0.1))

    datasets: List[Dict[str, str]] = []
    for entry in cfg.get("datasets", []):
        missing = [k for k in ("name", "gbff") if k not in entry]
        if missing:
            raise SystemExit(f"[error] dataset entry missing keys {missing}: {entry}")
        ds_entry = _make_dataset_entry(entry, block_size)
        if not Path(ds_entry["gbff"]).exists():
            raise SystemExit(f"[error] GBFF not found: {ds_entry['gbff']}")
        datasets.append(ds_entry)

    for spec in args.extra_dataset:
        entry = _parse_extra_dataset(spec)
        if not Path(entry["gbff"]).exists():
            raise SystemExit(f"[error] GBFF not found: {entry['gbff']}")
        datasets.append(_make_dataset_entry(entry, block_size))

    if not datasets:
        raise SystemExit("[error] No datasets specified (config + CLI empty).")

    _write_manifest(run_dir, datasets, block_size, windows_per_seq, val_frac, test_frac, args.force)

    for ds in datasets:
        _ensure_dataset(ds, windows_per_seq, val_frac, test_frac, block_size, args.force)

    combined_dir = Path("data/processed/combined") / args.run_id
    combined_dir.mkdir(parents=True, exist_ok=True)

    train_out = combined_dir / f"train_bs{block_size}.npz"
    val_out = combined_dir / f"val_bs{block_size}.npz"
    test_out = combined_dir / f"test_bs{block_size}.npz"

    _stack_npz([ds["train"] for ds in datasets], train_out)
    _stack_npz([ds["val"] for ds in datasets], val_out)
    _stack_npz([ds["test"] for ds in datasets], test_out)

    combined_manifest = {
        "train": str(train_out),
        "val": str(val_out),
        "test": str(test_out),
        "datasets": datasets,
    }
    combined_manifest_path = combined_dir / "manifest.json"
    combined_manifest_path.write_text(json.dumps(combined_manifest, indent=2))
    (run_dir / "combined_manifest.json").write_text(json.dumps(combined_manifest, indent=2))

    result = {
        "train_npz": str(train_out),
        "val_npz": str(val_out),
        "test_npz": str(test_out),
        "primary_dna": datasets[0]["dna"],
        "combined_manifest": str(combined_manifest_path),
    }
    result_path = run_dir / "pipeline_prepare.json"
    result_path.write_text(json.dumps(result, indent=2))

    print(f"[prepare] wrote {result_path}")


if __name__ == "__main__":
    main()
