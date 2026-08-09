#!/usr/bin/env python3
"""Train the latent protein EBM through the model-agnostic engine."""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.protein_lm.config import ProteinClassifierConfig
from src.protein_lm.dataset import (
    LengthBucketBatchSampler,
    MultiTaskProteinDataset,
    collate_protein_batch,
)
from src.protein_lm.ebm import ProteinLatentEBM
from src.protein_lm.ebm_task import (
    ProteinEBMTask,
    corrupt_sequence as corrupt_sequence,
    decode_protein_ebm_checkpoint,
    make_protein_ebm_checkpoint_adapter,
)
from src.protein_lm.models_multi import MultiTaskProteinClassifier
from src.protein_lm.tokenizer import ProteinTokenizer
from src.training.engine import EngineConfig, TrainingEngine
from src.training.optimizers import build_optimizer
from src.training.run_lifecycle import TrainingRun, configuration_fingerprint
from src.training.runtime import default_device
from src.training.strategies import AccumulatedBackpropStrategy


def parse_args():
    parser = argparse.ArgumentParser(description="Train Protein Latent EBM.")
    parser.add_argument("--config", required=True, help="Path to critic config file")
    parser.add_argument(
        "--critic_ckpt",
        required=True,
        help="Path to pre-trained MultiTask critic checkpoint",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of EBM training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for EBM")
    parser.add_argument(
        "--pooling",
        default="attention",
        help="Backbone pooling type (attention | mean)",
    )
    parser.add_argument("--family_dim", type=int, default=2000, help="Classifier family dimension")
    parser.add_argument("--function_dim", type=int, default=1000, help="Classifier function dimension")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden size of the EBM network")
    parser.add_argument(
        "--out_dir",
        default="runs/protein_ebm",
        help="Output directory for checkpoints",
    )
    parser.add_argument("--run_id", default=None, help="Run identifier")
    parser.add_argument("--resume", default=None, help="Checkpoint to resume")
    parser.add_argument("--seed", type=int, default=1337, help="RNG seed")
    return parser.parse_args()


def resolve_critic_checkpoint(path: str | Path):
    """Return critic model weights and optional architecture metadata."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError("critic checkpoint must contain a state mapping")
    spec = payload.get("model_spec")
    if "training_contract_version" in payload:
        state = payload.get("task", {}).get("model")
    elif "model_state_dict" in payload:
        state = payload["model_state_dict"]
    elif "model" in payload:
        state = payload["model"]
    elif payload and all(torch.is_tensor(value) for value in payload.values()):
        state = payload
    else:
        state = None
    if not isinstance(state, Mapping):
        raise ValueError("critic checkpoint has no recognized model state")
    return state, dict(spec or {})


class _EBMArtifacts:
    def __init__(self, curves_path: Path, epochs: int) -> None:
        self.curves_path = curves_path
        self.epochs = epochs
        self.training_metrics = {}

    def on_event(self, event) -> None:
        if event.name == "training_completed":
            self.training_metrics = dict(event.metrics)
        elif event.name == "epoch_completed":
            epoch = int(event.metadata["epoch"])
            train_loss = self.training_metrics["loss"].total
            val_loss = event.metrics["loss"].total
            with self.curves_path.open("a") as handle:
                handle.write(f"{epoch},{train_loss:.6f},{val_loss:.6f}\n")
            print(
                f"--- Epoch {epoch}/{self.epochs} Complete | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Energy Gap: {event.metrics['energy_gap'].total:.4f} ---",
                flush=True,
            )
        elif event.name == "checkpoint_saved":
            print(
                f"[saved] {event.metadata['filename']} "
                f"({event.metadata['reason']})",
                flush=True,
            )


def train_ebm(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = default_device()
    print(f"[ebm-train] using device: {device}")
    with open(args.config) as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, Mapping):
        raise TypeError("ProteinEBM configuration must be a YAML mapping")
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    if args.hidden_dim < 1:
        raise ValueError("hidden_dim must be positive")

    critic_path = Path(args.critic_ckpt).resolve()
    critic_state, critic_spec = resolve_critic_checkpoint(critic_path)
    task_dims = critic_spec.get(
        "task_dims",
        {
            "family": args.family_dim,
            "function": args.function_dim,
            "stability": 2,
        },
    )
    tokenizer = ProteinTokenizer()
    model_cfg = ProteinClassifierConfig(
        vocab_size=int(critic_spec.get("vocab_size", len(tokenizer))),
        n_layer=int(critic_spec.get("n_layer", cfg["n_layer"])),
        n_head=int(critic_spec.get("n_head", cfg["n_head"])),
        n_embd=int(critic_spec.get("n_embd", cfg["n_embd"])),
        block_size=int(critic_spec.get("block_size", cfg["block_size"])),
        dropout=float(critic_spec.get("dropout", cfg["dropout"])),
        pooling=str(critic_spec.get("pooling", args.pooling)),
        bidirectional=bool(critic_spec.get("bidirectional", True)),
        num_classes=2,
    )
    critic = MultiTaskProteinClassifier(model_cfg, task_dims)
    critic.load_state_dict(critic_state)
    critic.to(device)
    for parameter in critic.parameters():
        parameter.requires_grad = False
    critic.eval()

    print("[ebm-train] loading dataset paths from config")
    train_dataset = MultiTaskProteinDataset(
        cfg["train_data"], tokenizer, max_length=model_cfg.block_size
    )
    val_dataset = MultiTaskProteinDataset(
        cfg["val_data"], tokenizer, max_length=model_cfg.block_size
    )
    batch_size = int(cfg.get("batch_size", 4))
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=LengthBucketBatchSampler(
            train_dataset, batch_size=batch_size, shuffle=True
        ),
        collate_fn=lambda batch: collate_protein_batch(
            batch, pad_token_id=tokenizer.pad_token_id
        ),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=LengthBucketBatchSampler(
            val_dataset, batch_size=batch_size, shuffle=False
        ),
        collate_fn=lambda batch: collate_protein_batch(
            batch, pad_token_id=tokenizer.pad_token_id
        ),
    )

    model = ProteinLatentEBM(
        n_embd=model_cfg.n_embd,
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = build_optimizer(
        model.parameters(),
        {"lr": args.lr, "weight_decay": 0.01},
    )
    task = ProteinEBMTask(
        model=model,
        critic=critic,
        train_loader=train_loader,
        validation_loader=val_loader,
        tokenizer=tokenizer,
        device=device,
        block_size=model_cfg.block_size,
        mutation_rate=0.20,
        log_every_microbatches=50,
    )
    strategy = AccumulatedBackpropStrategy(optimizer, parameters=model.parameters())

    requested_dir = Path(args.out_dir)
    run_id = args.run_id or requested_dir.name
    fingerprint = configuration_fingerprint(
        {
            **cfg,
            "critic_ckpt": str(critic_path),
            "lr": args.lr,
            "pooling": model_cfg.pooling,
            "task_dims": task_dims,
            "hidden_dim": args.hidden_dim,
            "seed": args.seed,
        }
    )
    training_run = TrainingRun.open(
        requested_dir.parent,
        run_id,
        resume=args.resume,
        last_checkpoint_name="last_ebm.pt",
        target_epochs=args.epochs,
        config_fingerprint=fingerprint,
    )
    logger = training_run.logger()
    logger.__enter__()
    curves_path = training_run.scores / "curves.csv"
    if not curves_path.exists():
        curves_path.write_text("epoch,train_loss,val_loss\n")
    try:
        try:
            shutil.copy(args.config, training_run.checkpoints / "config.yaml")
        except OSError as exc:
            print(f"[ebm-train] warning: failed to copy config file: {exc}")
        model_spec = {
            "n_embd": model_cfg.n_embd,
            "hidden_dim": args.hidden_dim,
            "mutation_rate": 0.20,
            "critic_task_dims": dict(task_dims),
        }
        engine = TrainingEngine(
            task=task,
            strategy=strategy,
            run=training_run,
            config=EngineConfig(
                epochs=args.epochs,
                last_checkpoint_name="last_ebm.pt",
                best_checkpoint_name="best_ebm.pt",
                best_checkpoint_pattern="best_ebm_epoch_{epoch:03d}.pt",
                epoch_checkpoint_pattern="ebm_epoch_{epoch}.pt",
            ),
            device=device,
            callbacks=[_EBMArtifacts(curves_path, args.epochs)],
            run_fingerprint=fingerprint,
            checkpoint_decoder=decode_protein_ebm_checkpoint,
            checkpoint_payload_adapter=make_protein_ebm_checkpoint_adapter(
                critic_checkpoint=str(critic_path),
                model_spec=model_spec,
            ),
        )
        result = engine.fit()
        if result.status == "complete":
            summary_path = training_run.run_dir / "summary.md"
            summary_path.write_text(
                f"""# Run Summary: `{training_run.run_dir.name}`

## Status & Key Performance Indicators
- **Status:** Completed
- **Epochs Trained:** {args.epochs}
- **Best Epoch:** {engine.best_epoch}
- **Best Validation Loss:** {engine.best_metric:.4f}

## Model Architecture & Settings
- **Type:** Protein Latent Energy-Based Model (EBM)
- **Latent Dim:** {model_cfg.n_embd}
- **EBM Hidden Dim:** {args.hidden_dim}
- **Pooling:** {model_cfg.pooling}
- **Backbone Critic Checkpoint:** `{critic_path}`
"""
            )
            print(f"[summary] Wrote run summary to {summary_path}")
        return result
    finally:
        training_run.close()
        logger.__exit__(*sys.exc_info())


def main():
    return train_ebm(parse_args())


if __name__ == "__main__":
    main()
