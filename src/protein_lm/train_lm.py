import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from src.protein_lm.config import ProteinLMConfig, load_config
from src.protein_lm.data import create_dataloader
from src.protein_lm.models import ProteinConditionalTransformer
from src.protein_lm.tokenizer import ProteinTokenizer
from src.training.run_lifecycle import (
    TrainingRun,
    capture_rng_state,
    configuration_fingerprint,
    restore_rng_state,
)
from src.training.runtime import WallTimer, save_checkpoint_atomic


def train(config_path: str, resume: str | None = None, run_id: str | None = None):
    with open(config_path) as handle:
        config_data = yaml.safe_load(handle)
    lm_config = load_config(config_path, ProteinLMConfig)
    training_config = config_data.get("training", {})
    data_config = config_data.get("data", {})
    epochs = int(training_config["epochs"])
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    tokenizer = ProteinTokenizer()
    lm_config.vocab_size = len(tokenizer.vocab)
    seed = int(training_config.get("seed", 1337))
    train_generator = torch.Generator()
    num_workers = int(training_config.get("num_workers", (os.cpu_count() or 2) // 2))
    train_loader = create_dataloader(
        data_config["train_path"], training_config["batch_size"],
        num_workers=num_workers, tokenizer=tokenizer,
        block_size=lm_config.block_size, shuffle=True, generator=train_generator,
    )
    val_loader = create_dataloader(
        data_config["val_path"], training_config["batch_size"],
        num_workers=num_workers, tokenizer=tokenizer,
        block_size=lm_config.block_size, shuffle=False,
    )

    model = ProteinConditionalTransformer(lm_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training_config["lr"],
        weight_decay=training_config.get("weight_decay", 0.01),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    wall_timer = WallTimer(training_config.get("max_time_minutes"))
    optimizer_step = 0
    start_epoch = 0
    resume_microbatch = 0
    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        optimizer_step = int(checkpoint.get("optimizer_step", 0))
        complete = bool(checkpoint.get("epoch_complete", True))
        start_epoch = int(checkpoint["epoch"]) + (1 if complete else 0)
        resume_microbatch = 0 if complete else int(checkpoint.get("microbatch_idx", 0))
        restore_rng_state(checkpoint.get("rng_state"))

    current_microbatch = 0

    def save_checkpoint(path: Path, epoch: int, loss: float, reason: str) -> None:
        complete = reason == "epoch"
        save_checkpoint_atomic(
            {
                "epoch": epoch,
                "epoch_complete": complete,
                "microbatch_idx": 0 if complete else current_microbatch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": loss,
                "optimizer_step": optimizer_step,
                "checkpoint_reason": reason,
                "cfg": config_data,
                "run_fingerprint": fingerprint,
                "rng_state": capture_rng_state(),
                "run_progress": {
                    "completed_epochs": epoch + 1 if complete else epoch,
                    "current_epoch": epoch + 1,
                    "microbatch": 0 if complete else current_microbatch,
                    "optimizer_step": optimizer_step,
                },
            },
            path,
        )

    grad_accum = int(training_config.get("grad_accum_steps", 1))
    for epoch in range(start_epoch, epochs):
        train_generator.manual_seed(seed + epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for index, batch in enumerate(train_loader):
            if epoch == start_epoch and index < resume_microbatch:
                continue
            current_microbatch = index + 1
            input_ids = batch.to(device)
            targets = input_ids[:, 1:].contiguous()
            logits = model(input_ids[:, :-1]).contiguous()
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            (loss / grad_accum).backward()
            boundary = (index + 1) % grad_accum == 0 or index + 1 == len(train_loader)
            if boundary:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                every = int(training_config.get("checkpoint_every_steps", 0) or 0)
                if every and optimizer_step % every == 0:
                    save_checkpoint(training_run.checkpoints / "last.pt", epoch, float("inf"), "periodic")
            if index % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Step {index}, Loss: {loss.item():.4f}")
            if wall_timer.expired():
                save_checkpoint(training_run.checkpoints / "last.pt", epoch, float("inf"), "wall_time")
                training_run.close()
                return
        scheduler.step()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch.to(device)
                targets = input_ids[:, 1:].contiguous()
                logits = model(input_ids[:, :-1]).contiguous()
                val_loss += criterion(logits.view(-1, logits.size(-1)), targets.view(-1)).item()
        val_loss /= len(val_loader)
        print(f"Epoch {epoch + 1}, Val Loss: {val_loss:.4f}")
        save_checkpoint(training_run.checkpoints / f"epoch_{epoch + 1:03d}.pt", epoch, val_loss, "epoch")
        save_checkpoint(training_run.checkpoints / "last.pt", epoch, val_loss, "epoch")
    training_run.mark_complete({"completed_epochs": epochs})
    training_run.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a protein language model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    train(args.config, resume=args.resume, run_id=args.run_id)
