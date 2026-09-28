#!/usr/bin/env python3
"""Pretrain a DNA-shape encoder and optionally smoke-test CodonLM fusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from src.codonlm.biophysics import NucleotideEncoder, generate_shape_training_data
from src.codonlm.biophysics_task import BiophysicsEncoderTask
from src.codonlm.codon_tokenize import itos as CODON_ITOS
from src.codonlm.model_tiny_gpt import TinyGPT
from src.training.engine import EngineConfig, TrainingEngine
from src.training.run_lifecycle import TrainingRun, configuration_fingerprint
from src.training.runtime import (
    PeriodicCheckpointPolicy,
    WallTimer,
    default_device,
    save_artifact_atomic,
)
from src.training.strategies import AccumulatedBackpropStrategy


def build_one_hot_lookup(itos: list, device: torch.device) -> torch.Tensor:
    """Map token IDs to their three-position nucleotide one-hot encoding."""
    lookup = torch.zeros(len(itos), 3, 4, device=device)
    base_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    for token_id, token in enumerate(itos):
        if len(token) == 3 and all(base in base_to_idx for base in token):
            for position, base in enumerate(token):
                lookup[token_id, position, base_to_idx[base]] = 1.0
        elif len(token) == 1 and token in base_to_idx:
            lookup[token_id, 0, base_to_idx[token]] = 1.0
    return lookup


class _BiophysicsArtifacts:
    def __init__(self, *, task, encoder_path, curves_path, epochs):
        self.task = task
        self.encoder_path = encoder_path
        self.curves_path = curves_path
        self.epochs = epochs

    def on_event(self, event) -> None:
        if event.name != "epoch_completed":
            return
        epoch = int(event.metadata["epoch"])
        train_loss = event.metadata["training_metrics"]["loss"].total
        validation_loss = event.metrics["loss"].total
        with self.curves_path.open("a") as handle:
            handle.write(f"{epoch},{train_loss:.6f},{validation_loss:.6f}\n")
        print(
            f"Epoch {epoch:03d}/{self.epochs:03d} | Train Loss: {train_loss:.5f} | "
            f"Val Loss: {validation_loss:.5f}",
            flush=True,
        )
        if epoch == self.epochs:
            self.task.restore_best_model()
            save_artifact_atomic(
                dict(self.task.model.state_dict()), self.encoder_path
            )
            print(f"[success] Saved selected encoder to {self.encoder_path}")


def _resolve_generator_run(path: str | Path):
    run_dir = Path(path)
    checkpoint = run_dir / "checkpoints" / "best.pt"
    if not checkpoint.is_file():
        checkpoint = run_dir / "best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"generator checkpoint not found below {run_dir}")
    itos_path = run_dir / "itos.txt"
    itos = (
        [line.strip() for line in itos_path.read_text().splitlines() if line.strip()]
        if itos_path.is_file()
        else CODON_ITOS
    )
    return checkpoint, itos


def validate_fusion(encoder, generator_run, device):
    """Verify that encoder output can be injected into a shape-guided generator."""
    checkpoint_path, itos = _resolve_generator_run(generator_run)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    config = checkpoint.get("cfg", {})
    generator = TinyGPT(
        vocab_size=len(itos),
        block_size=int(config.get("block_size", 256)),
        n_layer=int(config.get("n_layer", 2)),
        n_head=int(config.get("n_head", 4)),
        n_embd=int(config.get("n_embd", 128)),
        dropout=float(config.get("dropout", 0.1)),
        use_shape_guidance=True,
    ).to(device)
    current = generator.state_dict()
    compatible = {
        name: value
        for name, value in state.items()
        if name in current and current[name].shape == value.shape
    }
    generator.load_state_dict(compatible, strict=False)
    lookup = build_one_hot_lookup(itos, device)
    generator.eval()
    encoder.eval()
    batch_size = min(4, len(itos))
    context_length = min(32, generator.block_size)
    tokens = torch.randint(0, len(itos), (batch_size, context_length), device=device)
    with torch.no_grad():
        one_hot = lookup[tokens].reshape(batch_size, context_length * 3, 4)
        shapes = encoder(one_hot)
        logits, _ = generator(tokens, shape_embeddings=shapes)
    if logits.shape != (batch_size, context_length, len(itos)):
        raise RuntimeError(f"unexpected fusion logits shape: {tuple(logits.shape)}")
    print(f"[success] Fusion smoke test passed: logits={tuple(logits.shape)}")


def train_encoder(
    *,
    config_path="configs/biophysics_encoder.yaml",
    out_dir="runs/biophysics_encoder",
    run_id=None,
    resume=None,
    fork_from=None,
    epochs=None,
    batch_size=None,
    learning_rate=None,
    total_samples=None,
    sequence_codons=None,
    seed=None,
    device_name=None,
    generator_run=None,
    max_time_minutes=None,
    checkpoint_every_steps=0,
):
    with open(config_path) as handle:
        source_config = yaml.safe_load(handle)
    if not isinstance(source_config, dict):
        raise TypeError("biophysics encoder config must be a mapping")
    epochs = int(source_config["epochs"] if epochs is None else epochs)
    batch_size = int(source_config["batch_size"] if batch_size is None else batch_size)
    learning_rate = float(
        source_config["learning_rate"] if learning_rate is None else learning_rate
    )
    total_samples = int(
        source_config["total_samples"] if total_samples is None else total_samples
    )
    sequence_codons = int(
        source_config["sequence_codons"] if sequence_codons is None else sequence_codons
    )
    seed = int(source_config["seed"] if seed is None else seed)
    run_id = run_id or source_config.get("run_id", "shape-encoder")
    model_config = {"d_shape": int(source_config.get("d_shape", 3))}
    if model_config["d_shape"] != 3:
        raise ValueError("d_shape must remain 3 for compatibility with CodonLM fusion")
    split_fractions = source_config.get(
        "split_fractions", {"train": 0.8, "validation": 0.1, "test": 0.1}
    )
    if set(split_fractions) != {"train", "validation", "test"}:
        raise ValueError("split_fractions must define train, validation, and test")
    fractions = [
        float(split_fractions[name]) for name in ("train", "validation", "test")
    ]
    if any(fraction <= 0 for fraction in fractions) or abs(sum(fractions) - 1) > 1e-8:
        raise ValueError("split fractions must be positive and sum to 1")
    for name, value in {
        "epochs": epochs,
        "batch_size": batch_size,
        "total_samples": total_samples,
        "sequence_codons": sequence_codons,
    }.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    n_train = int(total_samples * fractions[0])
    n_validation = int(total_samples * fractions[1])
    n_test = total_samples - n_train - n_validation
    if min(n_train, n_validation, n_test) < 1:
        raise ValueError(
            "total_samples is too small for the configured split fractions"
        )
    resolved_config = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "total_samples": total_samples,
        "split_fractions": dict(split_fractions),
        "split_counts": {"train": n_train, "validation": n_validation, "test": n_test},
        "sequence_codons": sequence_codons,
        "seed": seed,
        "model": model_config,
    }
    fingerprint = configuration_fingerprint(resolved_config)
    run = TrainingRun.open(
        out_dir,
        run_id,
        resume=resume,
        fork_from=fork_from,
        target_epochs=epochs,
        config_fingerprint=fingerprint,
    )
    run.start_logging()
    try:
        device = torch.device(device_name) if device_name else default_device()
        torch.manual_seed(seed)
        all_x, all_y = generate_shape_training_data(
            total_samples, sequence_codons, seed=seed
        )
        train_x, train_y = all_x[:n_train], all_y[:n_train]
        val_x = all_x[n_train : n_train + n_validation]
        val_y = all_y[n_train : n_train + n_validation]
        test_x = all_x[n_train + n_validation :]
        test_y = all_y[n_train + n_validation :]
        generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(
            TensorDataset(train_x, train_y),
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )
        val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=batch_size)
        test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=batch_size)
        encoder = NucleotideEncoder(**model_config).to(device)
        optimizer = torch.optim.AdamW(encoder.parameters(), lr=learning_rate)
        task = BiophysicsEncoderTask(
            model=encoder,
            train_loader=train_loader,
            validation_loader=val_loader,
            device=device,
            train_generator=generator,
            seed=seed,
        )
        (run.run_dir / "config.resolved.json").write_text(
            json.dumps(resolved_config, indent=2, sort_keys=True) + "\n"
        )
        curves_path = run.scores / "curves.csv"
        if not curves_path.exists():
            curves_path.write_text("epoch,train_loss,val_loss\n")
        engine = TrainingEngine(
            task=task,
            strategy=AccumulatedBackpropStrategy(
                optimizer, parameters=encoder.parameters()
            ),
            run=run,
            config=EngineConfig(epochs=epochs),
            device=device,
            callbacks=[
                _BiophysicsArtifacts(
                    task=task,
                    encoder_path=run.checkpoints / "biophysics_encoder.pt",
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
            encoder.eval()
            total_loss = 0.0
            total_weight = 0
            with torch.no_grad():
                for one_hot, targets in test_loader:
                    one_hot, targets = one_hot.to(device), targets.to(device)
                    weight = one_hot.size(0)
                    total_loss += (
                        float(torch.nn.functional.mse_loss(encoder(one_hot), targets))
                        * weight
                    )
                    total_weight += weight
            test_loss = total_loss / total_weight
            (run.scores / "test_metrics.json").write_text(
                json.dumps({"test_mse": test_loss, "test_samples": n_test}, indent=2)
                + "\n"
            )
            print(f"[test] held-out synthetic DNA-shape MSE: {test_loss:.6f}")
            if generator_run:
                validate_fusion(encoder, generator_run, device)
        return result
    finally:
        run.close()


def main():
    parser = argparse.ArgumentParser(
        description="Pretrain the DNA-shape encoder and optionally test CodonLM fusion."
    )
    parser.add_argument("--config", default="configs/biophysics_encoder.yaml")
    parser.add_argument("--out-dir", default="runs/biophysics_encoder")
    parser.add_argument("--run-id")
    parser.add_argument("--resume")
    parser.add_argument("--fork-from")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--total-samples", type=int)
    parser.add_argument("--sequence-codons", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--generator-run")
    parser.add_argument("--max-time-minutes", type=float)
    parser.add_argument("--checkpoint-every-steps", type=int, default=0)
    args = parser.parse_args()
    train_encoder(
        config_path=args.config,
        out_dir=args.out_dir,
        run_id=args.run_id,
        resume=args.resume,
        fork_from=args.fork_from,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        total_samples=args.total_samples,
        sequence_codons=args.sequence_codons,
        seed=args.seed,
        device_name=args.device,
        generator_run=args.generator_run,
        max_time_minutes=args.max_time_minutes,
        checkpoint_every_steps=args.checkpoint_every_steps,
    )


if __name__ == "__main__":
    main()
