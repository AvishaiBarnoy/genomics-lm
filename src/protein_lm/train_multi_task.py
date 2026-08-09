import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import json
import argparse
import yaml
import hashlib
import random
import csv
import sys
from pathlib import Path
from src.protein_lm.tokenizer import ProteinTokenizer
from src.protein_lm.models_multi import MultiTaskProteinClassifier
from src.protein_lm.config import ProteinClassifierConfig
from src.protein_lm.dataset import (
    MultiTaskProteinDataset,
    LengthBucketBatchSampler,
    collate_protein_batch,
)
from src.training.engine import EngineConfig, TrainingEngine
from src.training.optimizers import build_optimizer
from src.training.runtime import PeriodicCheckpointPolicy, WallTimer
from src.training.run_lifecycle import (
    TrainingRun,
    configuration_fingerprint,
)
from src.training.strategies import AccumulatedBackpropStrategy
from src.protein_lm.critic_task import (
    ProteinCriticTask,
    decode_protein_critic_checkpoint,
    make_protein_critic_checkpoint_adapter,
)


def load_compatible_model_weights(model, checkpoint_path, map_location="cpu"):
    """Load matching checkpoint tensors and skip incompatible task heads."""
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    source_state = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    target_state = model.state_dict()

    compatible = {}
    skipped = []
    for name, tensor in source_state.items():
        if name in target_state and target_state[name].shape == tensor.shape:
            compatible[name] = tensor
        else:
            skipped.append(name)

    target_state.update(compatible)
    model.load_state_dict(target_state)
    return len(compatible), skipped


def compute_multi_label_pos_weight(dataset, task, max_weight=100.0):
    labels = []
    for sample in dataset.samples:
        values = sample.get(task)
        if values is None:
            values = sample.get(f"{task}_labels")
        if values is None:
            continue
        if isinstance(values, dict):
            values = list(values.values())
        labels.append(torch.tensor(values, dtype=torch.float32))
    if not labels:
        raise ValueError(f"No labels found for multi-label task: {task}")
    matrix = torch.stack(labels)
    positives = matrix.sum(dim=0)
    negatives = matrix.shape[0] - positives
    weights = torch.where(
        positives > 0, negatives / positives.clamp_min(1.0), torch.ones_like(positives)
    )
    return weights.clamp(min=1.0, max=float(max_weight))


def compute_classification_class_weight(
    dataset,
    task: str,
    num_classes: int,
    mode: str = "sqrt_inverse_frequency",
    max_weight: float = 4.0,
) -> torch.Tensor:
    """Compute class weights from training labels only."""
    sample_fields = {
        "family": "pfam_id",
        "function": "ec_id",
        "stability": "stability_id",
    }
    if task not in sample_fields:
        raise ValueError(f"Unsupported classification task: {task}")
    if mode not in {"inverse_frequency", "sqrt_inverse_frequency"}:
        raise ValueError(f"Unsupported classification class-weighting mode: {mode}")
    if num_classes < 1:
        raise ValueError("num_classes must be positive")
    if max_weight <= 0:
        raise ValueError("classification_class_weight_max must be positive")

    counts = torch.zeros(num_classes, dtype=torch.float32)
    field = sample_fields[task]
    for sample in dataset.samples:
        label = sample.get(field, -1)
        if label is None or int(label) == -1:
            continue
        label = int(label)
        if label < 0 or label >= num_classes:
            raise ValueError(
                f"Training label {label} for {task} is outside [0, {num_classes})"
            )
        counts[label] += 1

    missing = torch.nonzero(counts == 0, as_tuple=False).flatten().tolist()
    if missing:
        raise ValueError(
            f"Training split has no examples for {task} classes: {missing}"
        )

    weights = counts.sum() / (num_classes * counts)
    if mode == "sqrt_inverse_frequency":
        weights = weights.sqrt()
    weights = weights / weights.mean()
    return weights.clamp(max=float(max_weight))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bind_critic_dataset(cfg: dict, config_path: str) -> dict:
    manifest_value = cfg.get("dataset_manifest")
    if not manifest_value:
        return {"status": "legacy_unverified"}
    root = Path(config_path).resolve().parents[1]
    manifest_path = Path(manifest_value)
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    manifest = json.loads(manifest_path.read_text())
    configured = {
        "train": Path(cfg["train_data"]),
        "validation": Path(cfg["val_data"]),
        "test": Path(cfg["test_data"]),
        "task_vocabs": Path(cfg["task_vocabs"]),
    }
    verified = {}
    for role, configured_path in configured.items():
        if not configured_path.is_absolute():
            configured_path = root / configured_path
        configured_path = configured_path.resolve()
        artifact = manifest["artifacts"][role]
        artifact_path = Path(artifact["path"]).resolve()
        if configured_path != artifact_path:
            raise ValueError(f"configured {role} does not match dataset manifest")
        actual_hash = _sha256(configured_path)
        if actual_hash != artifact["sha256"]:
            raise ValueError(f"{role} SHA-256 does not match dataset manifest")
        verified[role] = {"path": str(configured_path), "sha256": actual_hash}
    return {
        "status": "manifest_verified",
        "manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
        "artifacts": verified,
        "protocol": manifest.get("protocol"),
    }


def task_losses(
    logits_dict: dict,
    batch: dict,
    classification_tasks: tuple[str, ...],
    regression_tasks: tuple[str, ...],
    criterion: nn.Module | dict[str, nn.Module],
) -> dict[str, torch.Tensor]:
    losses = {}
    for task in classification_tasks:
        targets = batch[task]
        valid = targets != -1
        if bool(valid.any()):
            task_criterion = criterion[task] if isinstance(criterion, dict) else criterion
            losses[task] = task_criterion(logits_dict[task][valid], targets[valid])
    for task in regression_tasks:
        targets = batch[task].float()
        valid = torch.isfinite(targets)
        if bool(valid.any()):
            predictions = logits_dict[task].squeeze(-1)
            losses[task] = nn.functional.smooth_l1_loss(
                predictions[valid], targets[valid]
            )
    return losses


def accumulation_group_size(
    step: int, loader_length: int, grad_accum_steps: int
) -> int:
    if grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be at least one")
    group_start = (step // grad_accum_steps) * grad_accum_steps
    return min(grad_accum_steps, loader_length - group_start)


def mps_memory_summary():
    if not hasattr(torch, "mps"):
        return ""
    current_allocated = getattr(torch.mps, "current_allocated_memory", None)
    driver_allocated = getattr(torch.mps, "driver_allocated_memory", None)
    parts = []
    if current_allocated:
        parts.append(f"mps_current_mb={current_allocated() / 1024 / 1024:.1f}")
    if driver_allocated:
        parts.append(f"mps_driver_mb={driver_allocated() / 1024 / 1024:.1f}")
    return " | " + " | ".join(parts) if parts else ""


def train_multi_task(
    config_path,
    resume_path=None,
    run_id=None,
    transfer_from=None,
    max_time_minutes=None,
):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    if max_time_minutes is not None:
        cfg["max_time_minutes"] = float(max_time_minutes)

    device_name = cfg.get(
        "device", "mps" if torch.backends.mps.is_available() else "cpu"
    )
    device = torch.device(device_name)
    print(f"[*] Using device: {device}", flush=True)

    tokenizer = ProteinTokenizer()
    seed = int(cfg.get("seed", 1337))
    random.seed(seed)
    torch.manual_seed(seed)
    dataset_provenance = bind_critic_dataset(cfg, config_path)
    if dataset_provenance["status"] == "manifest_verified":
        print(
            f"[*] Dataset manifest verified: {dataset_provenance['manifest']['path']}",
            flush=True,
        )

    # Load vocabs to get task dimensions. Keep the production default, but allow
    # tests and small smoke runs to provide a self-contained vocab file.
    task_vocabs_path = cfg.get(
        "task_vocabs",
        "data/processed/protein_lm/multitask/task_vocabs.json",
    )
    with open(task_vocabs_path, "r") as f:
        vocabs = json.load(f)

    regression_tasks = tuple(cfg.get("regression_tasks", []))
    classification_tasks = tuple(
        task
        for task in ("family", "function", "stability")
        if task not in regression_tasks
    )
    task_dims = {
        "family": len(vocabs["pfam"]),
        "function": len(vocabs["ec"]),
        "stability": 1 if "stability" in regression_tasks else len(vocabs["stability"]),
    }
    multi_label_tasks = list(cfg.get("multi_label_tasks", []))
    for task in multi_label_tasks:
        task_dims[task] = len(vocabs[task])
    print(f"[*] Task Dimensions: {task_dims}", flush=True)

    # Build Config
    model_cfg = ProteinClassifierConfig(
        vocab_size=len(tokenizer.vocab),
        block_size=cfg.get("block_size", 512),
        n_layer=cfg.get("n_layer", 4),
        n_head=cfg.get("n_head", 4),
        n_embd=cfg.get("n_embd", 128),
        dropout=cfg.get("dropout", 0.1),
        num_classes=0,  # Dummy value for multi-task backbone
        use_checkpoint=cfg.get("use_checkpoint", False),
        pooling=cfg.get("pooling", "mean"),
        bidirectional=cfg.get("bidirectional", True),
    )

    print("[*] Building model...", flush=True)
    model = MultiTaskProteinClassifier(model_cfg, task_dims).to(device)

    transfer_checkpoint = transfer_from or cfg.get("transfer_from")
    if transfer_checkpoint:
        if resume_path:
            raise ValueError("--transfer_from and --resume are mutually exclusive")
        transfer_checkpoint = Path(transfer_checkpoint)
        if not transfer_checkpoint.exists():
            raise FileNotFoundError(
                f"Transfer checkpoint not found: {transfer_checkpoint}"
            )
        loaded, skipped = load_compatible_model_weights(
            model, transfer_checkpoint, map_location=device
        )
        print(
            f"[*] Transferred {loaded} compatible tensors from {transfer_checkpoint}",
            flush=True,
        )
        if skipped:
            print(
                f"[*] Skipped {len(skipped)} incompatible tensors, typically task-specific heads",
                flush=True,
            )

    print("[*] Loading datasets...", flush=True)
    dynamic_padding = bool(cfg.get("dynamic_padding", False))
    train_ds = MultiTaskProteinDataset(
        cfg["train_data"],
        tokenizer,
        max_length=model_cfg.block_size,
        dynamic_padding=dynamic_padding,
        multi_label_tasks=multi_label_tasks,
    )
    val_ds = MultiTaskProteinDataset(
        cfg["val_data"],
        tokenizer,
        max_length=model_cfg.block_size,
        dynamic_padding=dynamic_padding,
        multi_label_tasks=multi_label_tasks,
    )

    if dynamic_padding:
        train_loader = DataLoader(
            train_ds,
            batch_sampler=LengthBucketBatchSampler(
                train_ds,
                cfg.get("batch_size", 8),
                shuffle=True,
                seed=seed,
            ),
            collate_fn=lambda batch: collate_protein_batch(
                batch, tokenizer.pad_token_id
            ),
        )
        val_loader = DataLoader(
            val_ds,
            batch_sampler=LengthBucketBatchSampler(
                val_ds, cfg.get("batch_size", 8), shuffle=False
            ),
            collate_fn=lambda batch: collate_protein_batch(
                batch, tokenizer.pad_token_id
            ),
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.get("batch_size", 8),
            shuffle=True,
            collate_fn=lambda batch: collate_protein_batch(
                batch, tokenizer.pad_token_id
            ),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.get("batch_size", 8),
            collate_fn=lambda batch: collate_protein_batch(
                batch, tokenizer.pad_token_id
            ),
        )
    print(
        f"[*] Dataset sizes: train={len(train_ds)} val={len(val_ds)} "
        f"train_batches={len(train_loader)} val_batches={len(val_loader)}",
        flush=True,
    )

    optimizer = build_optimizer(model.parameters(), cfg)

    # Class weights are derived only from the training split. Validation remains
    # unweighted so its loss describes the frozen held-out distribution.
    classification_weighting = cfg.get("classification_class_weighting", "none")
    classification_weight_max = float(
        cfg.get("classification_class_weight_max", 4.0)
    )
    train_classification_criteria = {}
    for task in classification_tasks:
        class_weight = None
        if classification_weighting != "none":
            class_weight = compute_classification_class_weight(
                train_ds,
                task,
                task_dims[task],
                mode=classification_weighting,
                max_weight=classification_weight_max,
            ).to(device)
            print(
                f"[*] Classification weight for {task}: "
                f"min={float(class_weight.min().cpu()):.4f} "
                f"max={float(class_weight.max().cpu()):.4f} "
                f"mean={float(class_weight.mean().cpu()):.4f}",
                flush=True,
            )
        train_classification_criteria[task] = nn.CrossEntropyLoss(
            weight=class_weight, ignore_index=-1
        )
    validation_classification_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    multi_label_criteria = {}
    pos_weight_cfg = cfg.get("multi_label_pos_weight")
    pos_weight_max = cfg.get("multi_label_pos_weight_max", 100.0)
    for task in multi_label_tasks:
        pos_weight = None
        if pos_weight_cfg == "auto":
            pos_weight = compute_multi_label_pos_weight(
                train_ds, task, max_weight=pos_weight_max
            ).to(device)
        elif isinstance(pos_weight_cfg, dict) and task in pos_weight_cfg:
            pos_weight = torch.tensor(
                pos_weight_cfg[task], dtype=torch.float32, device=device
            )
        if pos_weight is not None:
            print(
                f"[*] Multi-label pos_weight for {task}: "
                f"{[round(float(v), 4) for v in pos_weight.detach().cpu()]}",
                flush=True,
            )
        multi_label_criteria[task] = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    epochs = int(cfg.get("epochs", 5))
    run_fingerprint = configuration_fingerprint(cfg)
    if not run_id:
        run_id = cfg.get("run_id", None)
    if not run_id:
        from datetime import date

        today = date.today().strftime("%Y-%m-%d")
        tag = Path(config_path).stem
        run_id = f"{today}_{tag}_{model_cfg.n_layer}L{model_cfg.n_head}H_d{model_cfg.n_embd}_e{epochs}"
    training_run = TrainingRun.open(
        "runs",
        run_id,
        resume=resume_path,
        last_checkpoint_name="last_critic.pt",
        target_epochs=epochs,
        config_fingerprint=run_fingerprint,
    )
    run_id = training_run.run_dir.name
    cfg["run_id"] = run_id

    print("[*] Starting Multi-Task Training...", flush=True)
    grad_accum_steps = int(cfg.get("grad_accum_steps", 1))
    print(f"[*] Gradient accumulation steps: {grad_accum_steps}", flush=True)
    log_every_steps = cfg.get("log_every_steps", 100)
    checkpoint_every_steps = cfg.get("checkpoint_every_steps", 0)
    print(
        f"[*] Progress logging every {log_every_steps} steps; "
        f"step checkpoint every {checkpoint_every_steps or 'disabled'} steps",
        flush=True,
    )

    max_time_minutes = cfg.get("max_time_minutes", None)
    if max_time_minutes:
        print(f"[*] Wall-time limit configured: {max_time_minutes} minutes", flush=True)

    run_logger = training_run.logger()
    run_logger.__enter__()
    log_csv = training_run.scores / "curves.csv"
    if not log_csv.exists():
        with open(log_csv, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_loss"])
    model_spec = {
        "vocab_size": model_cfg.vocab_size,
        "block_size": model_cfg.block_size,
        "n_layer": model_cfg.n_layer,
        "n_head": model_cfg.n_head,
        "n_embd": model_cfg.n_embd,
        "dropout": model_cfg.dropout,
        "pooling": model_cfg.pooling,
        "bidirectional": model_cfg.bidirectional,
        "task_dims": task_dims,
        "regression_tasks": list(regression_tasks),
    }

    class CriticLogger:
        def __init__(self):
            self.started = time.perf_counter()
            self.interval_started = self.started
            self.units = {"sequences": 0, "residues": 0}
            self.train_metrics = {}
            self.epoch_started = self.started
            self.current_epoch = None

        def on_event(self, event):
            if event.name == "group_committed":
                if event.context.epoch != self.current_epoch:
                    self.current_epoch = event.context.epoch
                    self.epoch_started = time.perf_counter()
                for name, value in event.metadata.get("committed_units", {}).items():
                    self.units[name] = self.units.get(name, 0) + value
                microbatch = event.context.microbatch + 1
                crossed_log_boundary = (
                    log_every_steps
                    and microbatch % log_every_steps < event.context.group_size
                )
                if crossed_log_boundary:
                    elapsed = max(time.perf_counter() - self.interval_started, 1e-9)
                    loss = event.metrics.get("loss")
                    print(
                        f"[progress] epoch={event.context.epoch + 1}/{epochs} "
                        f"step={event.context.microbatch + 1}/{len(train_loader)} "
                        f"optimizer_step={event.context.optimizer_step + 1} "
                        f"recent_loss={loss.total if loss else float('nan'):.4f} "
                        f"lr={optimizer.param_groups[0]['lr']:.2e} "
                        f"seq_per_sec={self.units['sequences'] / elapsed:.2f} "
                        f"residues_per_sec={self.units['residues'] / elapsed:.0f}"
                        f"{mps_memory_summary() if device.type == 'mps' else ''}",
                        flush=True,
                    )
                    self.interval_started = time.perf_counter()
                    self.units = {"sequences": 0, "residues": 0}
            elif event.name == "training_completed":
                self.train_metrics = dict(event.metrics)
            elif event.name == "epoch_completed":
                train_loss = self.train_metrics.get("loss")
                val_loss = event.metrics.get("loss")
                epoch = int(event.metadata["epoch"])
                with open(log_csv, "a", newline="") as handle:
                    csv.writer(handle).writerow(
                        [epoch, f"{train_loss.total:.4f}", f"{val_loss.total:.4f}"]
                    )
                print(
                    f"Epoch {epoch}/{epochs} | Train Loss: {train_loss.total:.4f} | "
                    f"Val Loss: {val_loss.total:.4f}",
                    flush=True,
                )
                print(
                    f"[timing] epoch={epoch} "
                    f"wall_sec={time.perf_counter() - self.epoch_started:.2f}",
                    flush=True,
                )
            elif event.name == "checkpoint_saved":
                print(
                    f"[checkpoint] saved {event.metadata['filename']} "
                    f"reason={event.metadata['reason']}",
                    flush=True,
                )

    task = ProteinCriticTask(
        model=model,
        train_loader=train_loader,
        validation_loader=val_loader,
        device=device,
        classification_tasks=classification_tasks,
        regression_tasks=regression_tasks,
        multi_label_tasks=multi_label_tasks,
        train_classification_criteria=train_classification_criteria,
        validation_classification_criterion=validation_classification_criterion,
        multi_label_criteria=multi_label_criteria,
        saliency_regularizer_weight=cfg.get("saliency_regularizer_weight", 0.0),
    )
    strategy = AccumulatedBackpropStrategy(optimizer, parameters=model.parameters())
    try:
        engine = TrainingEngine(
            task=task,
            strategy=strategy,
            run=training_run,
            config=EngineConfig(
                epochs=epochs,
                grad_accum_steps=grad_accum_steps,
                last_checkpoint_name="last_critic.pt",
                best_checkpoint_name="best_critic.pt",
                best_checkpoint_pattern="best_critic_epoch_{epoch:03d}.pt",
            ),
            device=device,
            callbacks=[CriticLogger()],
            wall_timer=WallTimer(max_time_minutes),
            checkpoint_policy=PeriodicCheckpointPolicy(
                every_steps=int(cfg.get("checkpoint_every_steps", 0) or 0),
                every_minutes=float(cfg.get("checkpoint_every_minutes", 0.0) or 0.0),
            ),
            run_fingerprint=run_fingerprint,
            checkpoint_decoder=decode_protein_critic_checkpoint,
            checkpoint_payload_adapter=make_protein_critic_checkpoint_adapter(
                config=cfg,
                dataset_provenance=dataset_provenance,
                task_vocabs=vocabs,
                model_spec=model_spec,
            ),
        )
        return engine.fit()
    finally:
        training_run.close()
        run_logger.__exit__(*sys.exc_info())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--resume", default=None, help="Path to checkpoint file to resume from"
    )
    parser.add_argument(
        "--transfer_from",
        default=None,
        help="Checkpoint to partially initialize compatible weights from",
    )
    parser.add_argument("--run_id", default=None, help="Unique run id")
    parser.add_argument("--max_time_minutes", type=float, default=None)
    args = parser.parse_args()
    train_multi_task(
        args.config,
        args.resume,
        args.run_id,
        args.transfer_from,
        args.max_time_minutes,
    )
