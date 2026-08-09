"""Train the corrected standalone protein classifier with the shared engine."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn
import yaml

from src.protein_lm.classifier_task import (
    ProteinClassifierTask,
    decode_protein_classifier_checkpoint,
    make_protein_classifier_checkpoint_adapter,
)
from src.protein_lm.config import (
    ProteinClassifierConfig,
    load_config,
    validate_protein_classifier_config,
)
from src.protein_lm.data import ProteinClassificationDataset, create_dataloader
from src.protein_lm.models import ProteinClassifier
from src.protein_lm.tokenizer import ProteinTokenizer
from src.training.engine import EngineConfig, TrainingEngine
from src.training.optimizers import build_optimizer
from src.training.run_lifecycle import TrainingRun, configuration_fingerprint
from src.training.runtime import PeriodicCheckpointPolicy, WallTimer, default_device
from src.training.strategies import AccumulatedBackpropStrategy


class _ClassifierConsole:
    def on_event(self, event) -> None:
        if event.name != "validation_completed":
            return
        print(
            f"Validation Loss: {event.metrics['loss'].total:.4f}, "
            f"Accuracy: {event.metrics['accuracy'].total:.4f}, "
            f"F1: {event.metrics['weighted_f1'].total:.4f}",
            flush=True,
        )


def train_classifier(
    config_path: str, resume: str | None = None, run_id: str | None = None
):
    with open(config_path) as handle:
        config_data = yaml.safe_load(handle)
    if not isinstance(config_data, Mapping):
        raise TypeError("Protein classifier configuration must be a YAML mapping")

    tokenizer = ProteinTokenizer()
    classifier_config = load_config(
        config_path,
        ProteinClassifierConfig,
        overrides={"vocab_size": len(tokenizer.vocab)},
    )
    training_config = config_data.get("training", {})
    data_config = config_data.get("data", {})
    validate_protein_classifier_config(config_data, classifier_config, tokenizer)

    epochs = training_config["epochs"]
    requested_run_id = run_id or config_data.get("run_id") or Path(config_path).stem
    fingerprint = configuration_fingerprint(config_data)
    device = default_device()
    seed = training_config.get("seed", 1337)
    torch.manual_seed(seed)
    train_generator = torch.Generator()
    label_field = data_config.get("label_field", "func_label")
    num_workers = int(training_config.get("num_workers", (os.cpu_count() or 2) // 2))

    train_loader = create_dataloader(
        data_config["train_path"],
        training_config["batch_size"],
        num_workers=num_workers,
        tokenizer=tokenizer,
        block_size=classifier_config.block_size,
        shuffle=True,
        dataset_class=ProteinClassificationDataset,
        label_field=label_field,
        generator=train_generator,
    )
    val_loader = create_dataloader(
        data_config["val_path"],
        training_config["batch_size"],
        num_workers=num_workers,
        tokenizer=tokenizer,
        block_size=classifier_config.block_size,
        shuffle=False,
        dataset_class=ProteinClassificationDataset,
        label_field=label_field,
        label_map=train_loader.dataset.label_map,
    )
    detected_classes = len(train_loader.dataset.label_map)
    if classifier_config.num_classes != detected_classes:
        raise ValueError(
            f"Configured num_classes={classifier_config.num_classes}, but the training "
            f"label map contains {detected_classes} classes"
        )

    model = ProteinClassifier(classifier_config).to(device)
    optimizer = build_optimizer(model.parameters(), training_config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    task = ProteinClassifierTask(
        model=model,
        train_loader=train_loader,
        validation_loader=val_loader,
        criterion=nn.CrossEntropyLoss(),
        device=device,
        train_generator=train_generator,
        seed=seed,
        label_map=train_loader.dataset.label_map,
        pad_token_id=tokenizer.pad_token_id,
        log_every_microbatches=training_config.get("log_every_steps", 100),
    )
    strategy = AccumulatedBackpropStrategy(
        optimizer,
        scheduler=scheduler,
        parameters=model.parameters(),
        grad_clip_norm=training_config.get("grad_clip_norm"),
        scheduler_interval="epoch",
    )
    training_run = TrainingRun.open(
        Path("runs") / "protein_classifier",
        requested_run_id,
        resume=resume,
        target_epochs=epochs,
        config_fingerprint=fingerprint,
    )
    logger = training_run.logger()
    logger.__enter__()
    try:
        print(f"Using device: {device}")
        engine = TrainingEngine(
            task=task,
            strategy=strategy,
            run=training_run,
            config=EngineConfig(
                epochs=epochs,
                grad_accum_steps=training_config.get("grad_accum_steps", 1),
                epoch_checkpoint_pattern="epoch_{epoch:03d}.pt",
            ),
            device=device,
            callbacks=[_ClassifierConsole()],
            wall_timer=WallTimer(training_config.get("max_time_minutes")),
            checkpoint_policy=PeriodicCheckpointPolicy(
                every_steps=training_config.get("checkpoint_every_steps", 0),
                every_minutes=training_config.get("checkpoint_every_minutes", 0.0),
            ),
            run_fingerprint=fingerprint,
            checkpoint_decoder=decode_protein_classifier_checkpoint,
            checkpoint_payload_adapter=make_protein_classifier_checkpoint_adapter(
                config_data
            ),
        )
        return engine.fit()
    finally:
        training_run.close()
        logger.__exit__(*sys.exc_info())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a protein classifier.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    train_classifier(args.config, resume=args.resume, run_id=args.run_id)
