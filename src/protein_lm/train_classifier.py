import torch
import torch.nn as nn
import yaml
import argparse
from pathlib import Path
import os
from sklearn.metrics import accuracy_score, f1_score

from src.protein_lm.config import ProteinClassifierConfig, load_config
from src.protein_lm.tokenizer import ProteinTokenizer
from src.protein_lm.models import ProteinClassifier
from src.protein_lm.data import create_dataloader, ProteinClassificationDataset
from src.training.runtime import WallTimer, save_checkpoint_atomic
from src.training.run_lifecycle import (
    TrainingRun,
    capture_rng_state,
    configuration_fingerprint,
    restore_rng_state,
)

def train_classifier(config_path: str, resume=None, run_id=None):
    """
    Trains the protein classifier.
    """
    # --- 1. Load Configuration ---
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    classifier_config = load_config(config_path, ProteinClassifierConfig)
    training_config = config_data.get('training', {})
    data_config = config_data.get('data', {})

    # --- 2. Setup ---
    epochs = int(training_config["epochs"])
    run_id = run_id or config_data.get("run_id") or Path(config_path).stem
    run_fingerprint = configuration_fingerprint(config_data)
    training_run = TrainingRun.open(
        Path("runs") / "protein_classifier",
        run_id,
        resume=resume,
        target_epochs=epochs,
        config_fingerprint=run_fingerprint,
    )
    run_logger = training_run.logger()
    run_logger.__enter__()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = ProteinTokenizer()
    classifier_config.vocab_size = len(tokenizer.vocab)
    seed = int(training_config.get("seed", 1337))
    train_generator = torch.Generator()

    # --- 3. Data Loading ---
    num_workers = int(training_config.get("num_workers", (os.cpu_count() or 2) // 2))
    train_loader = create_dataloader(
        data_config['train_path'],
        training_config['batch_size'],
        num_workers=num_workers,
        tokenizer=tokenizer,
        block_size=classifier_config.block_size,
        shuffle=True,
        dataset_class=ProteinClassificationDataset,
        label_field='func_label', # This assumes the label is in the 'func_label' field
        generator=train_generator,
    )
    val_loader = create_dataloader(
        data_config['val_path'],
        training_config['batch_size'],
        num_workers=num_workers,
        tokenizer=tokenizer,
        block_size=classifier_config.block_size,
        shuffle=False,
        dataset_class=ProteinClassificationDataset,
        label_field='func_label'
    )
    
    # Dynamically set num_classes from the dataset if not in config
    if not hasattr(classifier_config, 'num_classes') or classifier_config.num_classes is None:
        classifier_config.num_classes = len(train_loader.dataset.label_map)
        print(f"Number of classes detected from dataset: {classifier_config.num_classes}")


    # --- 4. Model Initialization ---
    model = ProteinClassifier(classifier_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config['lr'],
        weight_decay=training_config.get('weight_decay', 0.01)
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
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
        save_checkpoint_atomic({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
            'optimizer_step': optimizer_step,
            'checkpoint_reason': reason,
            'cfg': config_data,
            'epoch_complete': complete,
            'microbatch_idx': 0 if complete else current_microbatch,
            'run_fingerprint': run_fingerprint,
            'rng_state': capture_rng_state(),
            'run_progress': {
                'completed_epochs': epoch + 1 if complete else epoch,
                'current_epoch': epoch + 1,
                'microbatch': 0 if complete else current_microbatch,
                'optimizer_step': optimizer_step,
            },
        }, path)

    # --- 5. Training Loop ---
    grad_accum = int(training_config.get('grad_accum_steps', 1))
    for epoch in range(start_epoch, epochs):
        train_generator.manual_seed(seed + epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for i, (input_ids, labels) in enumerate(train_loader):
            if epoch == start_epoch and i < resume_microbatch:
                continue
            current_microbatch = i + 1
            input_ids, labels = input_ids.to(device), labels.to(device)

            logits = model(input_ids)
            loss = criterion(logits, labels)
            
            (loss / grad_accum).backward()

            boundary = (i + 1) % grad_accum == 0 or (i + 1) == len(train_loader)
            if boundary:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                every_steps = int(training_config.get("checkpoint_every_steps", 0) or 0)
                if every_steps > 0 and optimizer_step % every_steps == 0:
                    save_checkpoint(training_run.checkpoints / "last.pt", epoch, float("inf"), "periodic")

            if i % 100 == 0:
                print(f"Epoch {epoch+1}/{training_config['epochs']}, Step {i}, Loss: {loss.item():.4f}")
            if wall_timer.expired():
                save_checkpoint(training_run.checkpoints / "last.pt", epoch, float("inf"), "wall_time")
                print(f"[success] Wall-time reached; saved {training_run.checkpoints / 'last.pt'}.")
                training_run.close()
                return
        
        scheduler.step()

        # --- 6. Validation ---
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for input_ids, labels in val_loader:
                input_ids, labels = input_ids.to(device), labels.to(device)
                logits = model(input_ids)
                loss = criterion(logits, labels)
                val_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')

        print(f"Epoch {epoch+1}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

        # --- 7. Checkpointing ---
        save_checkpoint(training_run.checkpoints / f"epoch_{epoch+1:03d}.pt", epoch, val_loss, "epoch")
        save_checkpoint(training_run.checkpoints / "last.pt", epoch, val_loss, "epoch")

    training_run.mark_complete({"completed_epochs": epochs})
    training_run.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a protein classifier.")
    parser.add_argument("--config", type=str, required=True, help="Path to the config YAML file.")
    parser.add_argument("--resume")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    train_classifier(args.config, resume=args.resume, run_id=args.run_id)
