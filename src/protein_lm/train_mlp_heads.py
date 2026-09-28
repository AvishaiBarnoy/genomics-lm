#!/usr/bin/env python3
"""Train independent MLP feature heads through the shared training engine."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset

from src.protein_lm.mlp_heads_task import MLPHeadsTask
from src.training.engine import EngineConfig, TrainingEngine
from src.training.run_lifecycle import TrainingRun, configuration_fingerprint
from src.training.runtime import (
    PeriodicCheckpointPolicy,
    WallTimer,
    default_device,
    save_checkpoint_atomic,
)
from src.training.strategies import AccumulatedBackpropStrategy


class TaskFeatureDataset(Dataset):
    def __init__(self, npz_path, task_name, *, num_classes: int):
        with np.load(npz_path) as data:
            features = torch.as_tensor(data["X"], dtype=torch.float32)
            labels = torch.as_tensor(data[f"y_{task_name}"], dtype=torch.long)
        if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels):
            raise ValueError(f"invalid feature/label shapes for task {task_name!r}")
        mask = labels.ne(-1)
        self.X = features[mask]
        self.y = labels[mask]
        if len(self.y) and (int(self.y.min()) < 0 or int(self.y.max()) >= num_classes):
            raise ValueError(f"labels for task {task_name!r} exceed its vocabulary")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return {"X": self.X[index], "label": self.y[index]}


class MultiTaskMLPClassifier(nn.Module):
    def __init__(self, input_dim, task_dims):
        super().__init__()
        self.heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(input_dim, 256),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(256, dim),
                )
                for name, dim in task_dims.items()
            }
        )

    def forward(self, features):
        return {name: head(features) for name, head in self.heads.items()}


class _MLPHeadArtifacts:
    def __init__(self, *, task, path: Path, curves_path: Path, epochs: int) -> None:
        self.task = task
        self.path = path
        self.curves_path = curves_path
        self.epochs = epochs

    def on_event(self, event) -> None:
        if event.name != "epoch_completed":
            return
        epoch = int(event.metadata["epoch"])
        train = event.metadata["training_metrics"]
        validation = event.metrics
        summary = ", ".join(
            f"{name}: train={train[f'{name}_loss'].total:.4f} "
            f"val={validation[f'{name}_loss'].total:.4f}"
            for name in self.task.task_dims
        )
        row = [str(epoch)]
        for name in self.task.task_dims:
            row.extend(
                [
                    f"{train[f'{name}_loss'].total:.6f}",
                    f"{validation[f'{name}_loss'].total:.6f}",
                ]
            )
        with self.curves_path.open("a") as handle:
            handle.write(",".join(row) + "\n")
        print(f"Epoch {epoch:03d}/{self.epochs:03d} | {summary}", flush=True)
        if epoch == self.epochs:
            self.task.restore_best_heads()
            save_checkpoint_atomic(dict(self.task.model.state_dict()), self.path)
            print(f"[success] Saved selected MLP heads to {self.path}", flush=True)


def _load_task_dims(vocabs_json) -> dict[str, int]:
    with open(vocabs_json) as handle:
        vocabs = json.load(handle)
    required = {"pfam": "family", "ec": "function", "stability": "stability"}
    missing = set(required).difference(vocabs)
    if missing:
        raise ValueError(f"task vocabularies are missing: {sorted(missing)}")
    task_dims = {task: len(vocabs[source]) for source, task in required.items()}
    invalid = [name for name, size in task_dims.items() if size < 2]
    if invalid:
        raise ValueError(f"classification heads require at least two classes: {invalid}")
    return task_dims


def _make_loaders(path, task_dims, batch_size, *, shuffle, seed):
    loaders = {}
    generators = {}
    for name, dimension in task_dims.items():
        dataset = TaskFeatureDataset(path, name, num_classes=dimension)
        if len(dataset) == 0:
            raise ValueError(f"no labeled samples for task {name!r} in {path}")
        generator = torch.Generator()
        generator.manual_seed(seed)
        loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            generator=generator if shuffle else None,
        )
        generators[name] = generator
    return loaders, generators


def _evaluate(model, loaders, device):
    model.eval()
    with torch.no_grad():
        for name, loader in loaders.items():
            predictions, targets, top5, top10 = [], [], [], []
            for batch in loader:
                labels = batch["label"]
                logits = model.heads[name](batch["X"].to(device))
                predictions.extend(logits.argmax(dim=-1).cpu().tolist())
                targets.extend(labels.tolist())
                k = min(10, logits.size(-1))
                topk = torch.topk(logits, k=k, dim=-1).indices.cpu()
                correct = topk.eq(labels.unsqueeze(-1))
                top5.extend(correct[:, : min(5, k)].any(dim=-1).tolist())
                top10.extend(correct.any(dim=-1).tolist())
            print(f"Task: {name} | Top-1 Accuracy: {accuracy_score(targets, predictions):.4f}")
            if name in {"family", "function"}:
                print(
                    f"  Top-5 Accuracy: {sum(top5) / len(top5):.4f} | "
                    f"Top-10 Accuracy: {sum(top10) / len(top10):.4f}"
                )


def train_mlp_heads(
    train_npz,
    val_npz,
    vocabs_json,
    epochs=100,
    batch_size=64,
    lr=1e-3,
    out_dir="runs/protein_critic",
    *,
    run_id=None,
    resume=None,
    seed=1337,
    device_name=None,
    max_time_minutes=None,
    checkpoint_every_steps=0,
):
    for name, value in {"epochs": epochs, "batch_size": batch_size}.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(lr, bool) or not isinstance(lr, (int, float)) or lr <= 0:
        raise ValueError("lr must be positive")
    if checkpoint_every_steps < 0:
        raise ValueError("checkpoint_every_steps must be non-negative")
    if max_time_minutes is not None and max_time_minutes < 0:
        raise ValueError("max_time_minutes must be non-negative")
    task_dims = _load_task_dims(vocabs_json)
    with np.load(train_npz) as data:
        input_dim = int(data["X"].shape[1])
    fingerprint_data = {
        "train_npz": str(Path(train_npz).resolve()),
        "val_npz": str(Path(val_npz).resolve()),
        "vocabs_json": str(Path(vocabs_json).resolve()),
        "task_dims": task_dims,
        "input_dim": input_dim,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
    }
    fingerprint = configuration_fingerprint(fingerprint_data)
    training_run = TrainingRun.open(
        out_dir,
        run_id or "mlp-heads",
        resume=resume,
        target_epochs=epochs,
        config_fingerprint=fingerprint,
    )
    logger = training_run.logger()
    logger.__enter__()
    try:
        device = torch.device(device_name) if device_name else default_device()
        torch.manual_seed(seed)
        train_loaders, generators = _make_loaders(
            train_npz, task_dims, batch_size, shuffle=True, seed=seed
        )
        validation_loaders, _ = _make_loaders(
            val_npz, task_dims, batch_size, shuffle=False, seed=seed
        )
        model = MultiTaskMLPClassifier(input_dim, task_dims).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        task = MLPHeadsTask(
            model=model,
            train_loaders=train_loaders,
            validation_loaders=validation_loaders,
            device=device,
            train_generators=generators,
            seed=seed,
            task_dims=task_dims,
        )
        artifact_path = training_run.checkpoints / "mlp_heads.pt"
        curves_path = training_run.scores / "curves.csv"
        if not curves_path.exists():
            columns = ["epoch"]
            for name in task_dims:
                columns.extend([f"train_{name}_loss", f"val_{name}_loss"])
            curves_path.write_text(",".join(columns) + "\n")
        engine = TrainingEngine(
            task=task,
            strategy=AccumulatedBackpropStrategy(optimizer, parameters=model.parameters()),
            run=training_run,
            config=EngineConfig(epochs=epochs, grad_accum_steps=1),
            device=device,
            callbacks=[
                _MLPHeadArtifacts(
                    task=task,
                    path=artifact_path,
                    curves_path=curves_path,
                    epochs=epochs,
                )
            ],
            wall_timer=WallTimer(max_time_minutes),
            checkpoint_policy=PeriodicCheckpointPolicy(
                every_steps=checkpoint_every_steps
            ),
            run_fingerprint=fingerprint,
        )
        result = engine.fit()
        if result.status == "complete":
            _evaluate(model, validation_loaders, device)
        return result
    finally:
        training_run.close()
        logger.__exit__(*sys.exc_info())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_npz", required=True, help="Path to train features NPZ")
    parser.add_argument("--val_npz", required=True, help="Path to validation features NPZ")
    parser.add_argument(
        "--vocabs",
        default="data/processed/protein_lm/multitask/task_vocabs.json",
        help="Path to task vocabularies JSON",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out_dir", default="runs/protein_critic")
    parser.add_argument("--run_id")
    parser.add_argument("--resume")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device")
    parser.add_argument("--max_time_minutes", type=float)
    parser.add_argument("--checkpoint_every_steps", type=int, default=0)
    args = parser.parse_args()
    train_mlp_heads(
        args.train_npz,
        args.val_npz,
        args.vocabs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        out_dir=args.out_dir,
        run_id=args.run_id,
        resume=args.resume,
        seed=args.seed,
        device_name=args.device,
        max_time_minutes=args.max_time_minutes,
        checkpoint_every_steps=args.checkpoint_every_steps,
    )
