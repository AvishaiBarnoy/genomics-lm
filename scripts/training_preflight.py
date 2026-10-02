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
from src.training.contracts import MetricValue, StepOutput, TrainingPhase
from src.training.engine import EngineConfig, TrainingEngine
from src.training.run_lifecycle import TrainingRun, configuration_fingerprint
from src.training.strategies import AccumulatedBackpropStrategy


class _SharedEnginePreflightTask:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.model = torch.nn.Linear(1, 1, bias=False).to(device)

    def begin_phase(self, phase, epoch) -> None:
        self.model.train(phase == TrainingPhase.TRAIN)

    def end_phase(self, phase, epoch):
        return {}

    def train_batches(self, epoch):
        return [torch.tensor([[value]]) for value in (1.0, 2.0, 3.0, 4.0)]

    def validation_batches(self, epoch):
        return [torch.tensor([[1.0]])]

    def training_step(self, batch, context):
        loss = self.model(batch.to(self.device)).square().mean()
        return StepOutput(
            loss,
            {"loss": MetricValue(float(loss.detach().cpu()), 1.0)},
        )

    def validation_step(self, batch, context):
        loss = self.model(batch.to(self.device)).square().mean()
        return StepOutput(
            loss,
            {"loss": MetricValue(float(loss.detach().cpu()), 1.0)},
        )

    def state_dict(self):
        return {"model": self.model.state_dict()}

    def load_state_dict(self, state) -> None:
        self.model.load_state_dict(state["model"])


def _shared_engine(run: TrainingRun, device: torch.device, epochs: int):
    task = _SharedEnginePreflightTask(device)
    optimizer = torch.optim.SGD(task.model.parameters(), lr=0.05)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    fingerprint = configuration_fingerprint(
        {"task": "shared-engine-preflight", "optimizer": "sgd", "lr": 0.05}
    )
    engine = TrainingEngine(
        task=task,
        strategy=AccumulatedBackpropStrategy(
            optimizer,
            scheduler=scheduler,
            parameters=task.model.parameters(),
        ),
        run=run,
        config=EngineConfig(epochs=epochs, grad_accum_steps=2),
        device=device,
        run_fingerprint=fingerprint,
    )
    return engine, fingerprint


def _run_shared_engine_fork_preflight(root: Path, device_name: str) -> dict:
    device = torch.device(device_name)
    run_root = root / "shared-engine-runs"
    torch.manual_seed(1337)
    source = TrainingRun.open(run_root, "source")
    source_engine, fingerprint = _shared_engine(source, device, epochs=1)
    source_result = source_engine.fit()
    source_checkpoint = source.checkpoints / "best.pt"
    source.close()

    fork = TrainingRun.open(
        run_root,
        "fork",
        fork_from=source_checkpoint,
        target_epochs=2,
        config_fingerprint=fingerprint,
    )
    fork_engine, _ = _shared_engine(fork, device, epochs=2)
    fork_result = fork_engine.fit()
    lineage = json.loads((fork.run_dir / "run_lineage.json").read_text())
    fork.close()

    if source_result.state.optimizer_step != 2:
        raise RuntimeError("shared-engine source did not complete two optimizer steps")
    if fork_result.state.optimizer_step != 4:
        raise RuntimeError("shared-engine fork did not advance to four optimizer steps")
    if lineage["source_run_id"] != "source" or lineage["fork_run_id"] != "fork":
        raise RuntimeError(f"unexpected shared-engine fork lineage: {lineage}")
    return {
        "status": "passed",
        "device": device_name,
        "source_optimizer_steps": source_result.state.optimizer_step,
        "fork_optimizer_steps": fork_result.state.optimizer_step,
        "lineage": lineage,
    }


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


def _run_training(
    repo: Path,
    root: Path,
    config: Path,
    resume: Path | None = None,
    *,
    fork_from: Path | None = None,
    run_id: str = "corrected-preflight",
):
    command = [
        sys.executable, "-m", "src.codonlm.train_codon_lm", "--config", str(config),
        "--run_id", run_id,
    ]
    if resume is not None:
        command.extend(["--resume", str(resume)])
    if fork_from is not None:
        command.extend(["--fork-from", str(fork_from)])
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True)
    log_name = "fork.log" if fork_from else ("resume.log" if resume else "initial.log")
    (root / log_name).write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"training command failed; see {root / log_name}")
    return command


def _checkpoint_summary(path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "path": str(path.resolve()),
        "training_contract_version": checkpoint.get("training_contract_version"),
        "step": int(checkpoint["step"]),
        "epoch": int(checkpoint["epoch"]),
        "scheduler_last_epoch": int(checkpoint["scheduler"]["last_epoch"]),
        "consumed_train_tokens": int(checkpoint["consumed_train_tokens"]),
        "accumulation_health": checkpoint["accumulation_health"],
        "runtime_memory": checkpoint["runtime_memory"],
        "dataset_manifest": checkpoint["cfg"]["dataset_manifest"],
        "vocabulary_sha256": checkpoint["cfg"]["vocabulary"]["sha256"],
        "device": checkpoint["cfg"]["device"],
        "engine": checkpoint.get("engine"),
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
    source_best = root / "runs" / "corrected-preflight" / "checkpoints" / "best.pt"
    _run_training(
        repo,
        root,
        config,
        fork_from=source_best,
        run_id="corrected-preflight-fork",
    )
    codon_fork_checkpoint = (
        root / "runs" / "corrected-preflight-fork" / "checkpoints" / "last.pt"
    )
    codon_fork = _checkpoint_summary(codon_fork_checkpoint)
    resume_command = _run_training(repo, root, config, checkpoint)
    resumed = _checkpoint_summary(checkpoint)
    if resumed["step"] <= initial["step"]:
        raise RuntimeError("optimizer step did not advance after resume")
    if initial["training_contract_version"] != 1 or resumed["training_contract_version"] != 1:
        raise RuntimeError("CodonLM did not write shared-engine checkpoint contracts")
    if codon_fork["step"] != 4 or codon_fork["training_contract_version"] != 1:
        raise RuntimeError("CodonLM shared-engine fork did not restore and advance state")
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
    shared_engine_fork = _run_shared_engine_fork_preflight(root, args.device)
    if args.device == "mps":
        torch.mps.synchronize()
    report = {
        "status": "passed", "requested_device": args.device,
        "actual_device": resumed["device"], "dataset_id": manifest["dataset"]["id"],
        "dataset_schema": manifest["schema"], "initial": initial, "resumed": resumed,
        "commands": {"initial": initial_command, "resume": resume_command},
        "codon_engine_fork": codon_fork,
        "shared_engine_fork": shared_engine_fork,
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
