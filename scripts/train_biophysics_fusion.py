#!/usr/bin/env python3
"""Pretrain a DNA-shape encoder and optionally smoke-test CodonLM fusion."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
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
    save_checkpoint_atomic,
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
            save_checkpoint_atomic(
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
    tokens = torch.randint(0, len(itos), (4, 32), device=device)
    with torch.no_grad():
        shapes = encoder(lookup[tokens].view(4, 96, 4))
        logits, _ = generator(tokens, shape_embeddings=shapes)
    if logits.shape != (4, 32, len(itos)):
        raise RuntimeError(f"unexpected fusion logits shape: {tuple(logits.shape)}")
    print(f"[success] Fusion smoke test passed: logits={tuple(logits.shape)}")


def train_encoder(
    *,
    out_dir="runs/biophysics_encoder",
    run_id="shape-encoder",
    resume=None,
    epochs=5,
    batch_size=64,
    learning_rate=0.005,
    train_samples=8000,
    validation_samples=1000,
    sequence_codons=60,
    seed=1337,
    device_name=None,
    generator_run=None,
    max_time_minutes=None,
    checkpoint_every_steps=0,
):
    for name, value in {
        "epochs": epochs,
        "batch_size": batch_size,
        "train_samples": train_samples,
        "validation_samples": validation_samples,
        "sequence_codons": sequence_codons,
    }.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    config = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "train_samples": train_samples,
        "validation_samples": validation_samples,
        "sequence_codons": sequence_codons,
        "seed": seed,
    }
    fingerprint = configuration_fingerprint(config)
    run = TrainingRun.open(
        out_dir,
        run_id,
        resume=resume,
        target_epochs=epochs,
        config_fingerprint=fingerprint,
    )
    logger = run.logger()
    logger.__enter__()
    try:
        device = torch.device(device_name) if device_name else default_device()
        random.seed(seed)
        torch.manual_seed(seed)
        train_x, train_y = generate_shape_training_data(train_samples, sequence_codons)
        val_x, val_y = generate_shape_training_data(validation_samples, sequence_codons)
        generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(
            TensorDataset(train_x, train_y),
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )
        val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=batch_size)
        encoder = NucleotideEncoder(d_shape=3).to(device)
        optimizer = torch.optim.AdamW(encoder.parameters(), lr=learning_rate)
        task = BiophysicsEncoderTask(
            model=encoder,
            train_loader=train_loader,
            validation_loader=val_loader,
            device=device,
            train_generator=generator,
            seed=seed,
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
        if result.status == "complete" and generator_run:
            validate_fusion(encoder, generator_run, device)
        return result
    finally:
        run.close()
        logger.__exit__(*sys.exc_info())


def main():
    parser = argparse.ArgumentParser(
        description="Pretrain the DNA-shape encoder and optionally test CodonLM fusion."
    )
    parser.add_argument("--out-dir", default="runs/biophysics_encoder")
    parser.add_argument("--run-id", default="shape-encoder")
    parser.add_argument("--resume")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--train-samples", type=int, default=8000)
    parser.add_argument("--validation-samples", type=int, default=1000)
    parser.add_argument("--sequence-codons", type=int, default=60)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device")
    parser.add_argument("--generator-run")
    parser.add_argument("--max-time-minutes", type=float)
    parser.add_argument("--checkpoint-every-steps", type=int, default=0)
    args = parser.parse_args()
    train_encoder(
        out_dir=args.out_dir,
        run_id=args.run_id,
        resume=args.resume,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        train_samples=args.train_samples,
        validation_samples=args.validation_samples,
        sequence_codons=args.sequence_codons,
        seed=args.seed,
        device_name=args.device,
        generator_run=args.generator_run,
        max_time_minutes=args.max_time_minutes,
        checkpoint_every_steps=args.checkpoint_every_steps,
    )


if __name__ == "__main__":
    main()
