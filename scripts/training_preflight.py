#!/usr/bin/env python3
"""Run a corrected dataset -> train -> checkpoint -> resume preflight."""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from src.codonlm.dataset_manifest import (
    SCHEMA_NAME,
    SCHEMA_VERSION,
    artifact_entry,
    file_sha256,
    finalize_manifest,
    load_dataset_manifest,
)


def _write_fixture(root: Path) -> tuple[Path, dict]:
    data_dir = root / "dataset"
    data_dir.mkdir(parents=True, exist_ok=True)
    source = data_dir / "source.gbff"
    source.write_text("LOCUS preflight fixture\n")
    tokens = ["<PAD>", "<BOS_CDS>", "<EOS_CDS>", "<SEP>", "AAA", "CCC", "GGG", "TTT"]
    vocabulary = data_dir / "itos.txt"
    vocabulary.write_text("\n".join(tokens) + "\n")
    auxiliary = {
        "source_metadata": data_dir / "cds_meta.tsv",
        "source_dna": data_dir / "cds_dna.txt",
        "fragment_metadata": data_dir / "cds_fragments.tsv",
        "leakage_audit": data_dir / "leakage_audit.json",
    }
    for name, path in auxiliary.items():
        path.write_text(json.dumps({"fixture": name, "status": "passed"}) + "\n")
    artifacts = {
        "vocabulary": artifact_entry(vocabulary, data_dir, "vocabulary"),
        **{name: artifact_entry(path, data_dir, name) for name, path in auxiliary.items()},
    }
    split_rows = {"train": 5, "val": 2, "test": 2}
    for split, rows in split_rows.items():
        x = np.tile(np.array([1, 4, 5, 6, 7, 4, 5, 6], dtype=np.int32), (rows, 1))
        y = np.tile(np.array([4, 5, 6, 7, 4, 5, 6, 2], dtype=np.int32), (rows, 1))
        dataset = data_dir / f"{split}.npz"
        np.savez_compressed(dataset, X=x, Y=y)
        artifacts[f"{split}_tokens"] = artifact_entry(dataset, data_dir, f"{split}_tokens")
        packing = data_dir / f"{split}_packing.tsv"
        packing.write_text("window_index\n" + "\n".join(map(str, range(rows))) + "\n")
        artifacts[f"{split}_packing_metadata"] = artifact_entry(
            packing, data_dir, f"{split}_packing_metadata"
        )
    manifest = finalize_manifest(
        {
            "schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
            "dataset": {"id": "pending", "scientific_valid": True, "source_record_count": 9},
            "sources": {
                "preflight-genome": {
                    "path": str(source.resolve()),
                    "sha256": file_sha256(source),
                    "bytes": source.stat().st_size,
                    "identity_source": "preflight_fixture",
                }
            },
            "split_policy": {
                "effective_group_by": "genome",
                "allow_sequence_split": False,
                "scientific_valid": True,
                "requested_fractions": {"val": 2 / 9, "test": 2 / 9},
                "record_counts": split_rows,
                "groups_by_split": {
                    "train": ["train-genome"], "val": ["val-genome"], "test": ["test-genome"]
                },
            },
            "vocabulary": {
                "sha256": file_sha256(vocabulary),
                "size": len(tokens),
                "special_tokens": {"<PAD>": 0, "<BOS_CDS>": 1, "<EOS_CDS>": 2, "<SEP>": 3},
            },
            "leakage_audit": {
                "status": "passed", "homology_audit_skipped": False,
                "exact_duplicate_override": False,
            },
            "tokenization": {"ambiguous_codon_policy": {"name": "split"}},
            "packing": {"mode": "fixed", "transition_policy": "exactly_once"},
            "reproducibility": {"split_seed": 1337, "packing_seed": 1337},
            "artifacts": artifacts,
        }
    )
    manifest_path = data_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    load_dataset_manifest(manifest_path)
    return manifest_path, manifest


def _config(root: Path, manifest_path: Path, device: str, epochs: int) -> Path:
    data_dir = manifest_path.parent
    config = {
        "vocab_size": 8, "block_size": 8, "n_layer": 1, "n_head": 1,
        "n_embd": 16, "dropout": 0.1, "batch_size": 2,
        "grad_accum_steps": 2, "lr": 0.001, "min_lr": 0.0001,
        "weight_decay": 0.0, "warmup_steps": 0, "epochs": epochs,
        "optimizer": "adamw", "scheduler": "cosine", "amp": False,
        "use_checkpoint": False, "use_sdpa": True, "sep_mask_enabled": True,
        "early_stop_patience": 10, "seed": 1337, "num_workers": 0,
        "device": device, "dataset_manifest": str(manifest_path),
        "itos_path": str(data_dir / "itos.txt"),
        "train_npz": str(data_dir / "train.npz"),
        "val_npz": str(data_dir / "val.npz"),
        "test_npz": str(data_dir / "test.npz"),
        "out_dir": str(root / "unused-checkpoints"),
        "scores_dir": str(root / "unused-scores"),
    }
    config_path = root / "preflight.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=True))
    return config_path


def _run_training(repo: Path, root: Path, config: Path, resume: Path | None = None):
    command = [
        sys.executable, "-m", "src.codonlm.train_codon_lm", "--config", str(config),
        "--run_id", "corrected-preflight",
    ]
    if resume is not None:
        command.extend(["--resume", str(resume)])
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True)
    log_name = "resume.log" if resume else "initial.log"
    (root / log_name).write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"training command failed; see {root / log_name}")
    return command


def _checkpoint_summary(path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "path": str(path.resolve()),
        "step": int(checkpoint["step"]),
        "epoch": int(checkpoint["epoch"]),
        "scheduler_last_epoch": int(checkpoint["scheduler"]["last_epoch"]),
        "consumed_train_tokens": int(checkpoint["consumed_train_tokens"]),
        "accumulation_health": checkpoint["accumulation_health"],
        "runtime_memory": checkpoint["runtime_memory"],
        "dataset_manifest": checkpoint["cfg"]["dataset_manifest"],
        "vocabulary_sha256": checkpoint["cfg"]["vocabulary"]["sha256"],
        "device": checkpoint["cfg"]["device"],
    }


def _mps_memory():
    if not torch.backends.mps.is_available():
        return None
    driver = getattr(torch.mps, "driver_allocated_memory", None)
    return {
        "current_allocated_bytes": int(torch.mps.current_allocated_memory()),
        "driver_allocated_bytes": int(driver()) if callable(driver) else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "mps"), required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("--device mps requested but MPS is not available")
    root = args.work_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    started = time.perf_counter()
    memory_before = _mps_memory()
    manifest_path, manifest = _write_fixture(root)
    config = _config(root, manifest_path, args.device, epochs=1)
    initial_command = _run_training(repo, root, config)
    checkpoint = root / "runs" / "corrected-preflight" / "checkpoints" / "last.pt"
    initial = _checkpoint_summary(checkpoint)
    config = _config(root, manifest_path, args.device, epochs=2)
    resume_command = _run_training(repo, root, config, checkpoint)
    resumed = _checkpoint_summary(checkpoint)
    if resumed["step"] <= initial["step"]:
        raise RuntimeError("optimizer step did not advance after resume")
    if resumed["scheduler_last_epoch"] <= initial["scheduler_last_epoch"]:
        raise RuntimeError("scheduler did not advance after resume")
    if resumed["consumed_train_tokens"] <= initial["consumed_train_tokens"]:
        raise RuntimeError("committed non-PAD token count did not advance after resume")
    if resumed["dataset_manifest"]["dataset_id"] != manifest["dataset"]["id"]:
        raise RuntimeError("checkpoint dataset identity does not match fixture manifest")
    expected_health = {
        "active_microbatches": 0, "nonfinite_microbatches": 0,
        "aborted_groups": 0, "discarded_finite_microbatches": 0,
    }
    if resumed["accumulation_health"] != expected_health:
        raise RuntimeError(f"unexpected accumulation health: {resumed['accumulation_health']}")
    if args.device == "mps":
        torch.mps.synchronize()
    report = {
        "status": "passed", "requested_device": args.device,
        "actual_device": resumed["device"], "dataset_id": manifest["dataset"]["id"],
        "dataset_schema": manifest["schema"], "initial": initial, "resumed": resumed,
        "commands": {"initial": initial_command, "resume": resume_command},
        "wall_seconds": time.perf_counter() - started,
        "memory": {
            "mps_before": memory_before, "mps_after": _mps_memory(),
            "process_max_rss_raw": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "children_max_rss_raw": int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss),
        },
        "environment": {
            "python": sys.version, "pytorch": torch.__version__,
            "platform": platform.platform(), "mps_available": torch.backends.mps.is_available(),
        },
    }
    report_path = root / "preflight_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "passed", "report": str(report_path), "steps": resumed["step"]}))


if __name__ == "__main__":
    main()
