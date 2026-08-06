import argparse
import os
import sys
from pathlib import Path

import torch
import yaml

from src.protein_lm.config import ProteinLMConfig, load_config
from src.protein_lm.data import create_dataloader
from src.protein_lm.models import ProteinConditionalTransformer
from src.protein_lm.tasks import (
    ProteinLMTask,
    decode_protein_lm_checkpoint,
    protein_lm_checkpoint_adapter,
)
from src.protein_lm.tokenizer import ProteinTokenizer
from src.training.engine import EngineConfig, TrainingEngine
from src.training.run_lifecycle import TrainingRun, configuration_fingerprint
from src.training.runtime import PeriodicCheckpointPolicy, WallTimer, default_device
from src.training.strategies import AccumulatedBackpropStrategy, PrecisionPolicy


class _ProteinLMConsole:
    def on_event(self, event):
        if event.name == "validation_completed" and "loss" in event.metrics:
            print(f"Validation Loss: {event.metrics['loss'].total:.4f}")


def train(config_path: str, resume: str | None = None, run_id: str | None = None):
    with open(config_path) as handle:
        config_data = yaml.safe_load(handle)
    lm_config = load_config(config_path, ProteinLMConfig)
    training_config = config_data.get("training", {})
    data_config = config_data.get("data", {})
    epochs = training_config["epochs"]
    requested_run_id = run_id or config_data.get("run_id") or Path(config_path).stem
    fingerprint = configuration_fingerprint(config_data)
    training_run = TrainingRun.open(
        Path("runs") / "protein_lm",
        requested_run_id,
        resume=resume,
        target_epochs=epochs,
        config_fingerprint=fingerprint,
    )
    logger = training_run.logger()
    logger.__enter__()
    try:
        device = default_device()
        print(f"Using device: {device}")
        tokenizer = ProteinTokenizer()
        lm_config.vocab_size = len(tokenizer.vocab)
        seed = int(training_config.get("seed", 1337))
        torch.manual_seed(seed)
        train_generator = torch.Generator()
        num_workers = int(
            training_config.get("num_workers", (os.cpu_count() or 2) // 2)
        )
        train_loader = create_dataloader(
            data_config["train_path"],
            training_config["batch_size"],
            num_workers=num_workers,
            tokenizer=tokenizer,
            block_size=lm_config.block_size,
            shuffle=True,
            generator=train_generator,
        )
        val_loader = create_dataloader(
            data_config["val_path"],
            training_config["batch_size"],
            num_workers=num_workers,
            tokenizer=tokenizer,
            block_size=lm_config.block_size,
            shuffle=False,
        )
        model = ProteinConditionalTransformer(lm_config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_config["lr"],
            weight_decay=training_config.get("weight_decay", 0.01),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )
        task = ProteinLMTask(
            model=model,
            train_loader=train_loader,
            validation_loader=val_loader,
            tokenizer=tokenizer,
            device=device,
            train_generator=train_generator,
            seed=seed,
            log_every_microbatches=int(training_config.get("log_every_steps", 100)),
        )
        amp = bool(training_config.get("amp", False)) and device.type in {
            "cuda",
            "mps",
        }
        strategy = AccumulatedBackpropStrategy(
            optimizer,
            scheduler=scheduler,
            parameters=model.parameters(),
            grad_clip_norm=training_config.get("grad_clip_norm"),
            precision=PrecisionPolicy(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp,
                scale_gradients=amp and device.type == "cuda",
            ),
            scheduler_interval="epoch",
        )
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
            callbacks=[_ProteinLMConsole()],
            wall_timer=WallTimer(training_config.get("max_time_minutes")),
            checkpoint_policy=PeriodicCheckpointPolicy(
                every_steps=training_config.get("checkpoint_every_steps", 0),
                every_minutes=training_config.get("checkpoint_every_minutes", 0.0),
            ),
            run_fingerprint=fingerprint,
            checkpoint_decoder=decode_protein_lm_checkpoint,
            checkpoint_payload_adapter=protein_lm_checkpoint_adapter(config_data),
        )
        return engine.fit()
    finally:
        training_run.close()
        logger.__exit__(*sys.exc_info())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a protein language model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    train(args.config, resume=args.resume, run_id=args.run_id)
