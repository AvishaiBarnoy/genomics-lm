import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import json
import argparse
import yaml
import hashlib
import random
from pathlib import Path
from src.protein_lm.tokenizer import ProteinTokenizer
from src.protein_lm.models_multi import MultiTaskProteinClassifier
from src.protein_lm.config import ProteinClassifierConfig
from src.protein_lm.dataset import (
    MultiTaskProteinDataset,
    LengthBucketBatchSampler,
    collate_protein_batch,
)
from src.training.runtime import (
    PeriodicCheckpointPolicy,
    RunLogger,
    WallTimer,
    save_checkpoint_atomic,
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
    criterion: nn.Module,
) -> dict[str, torch.Tensor]:
    losses = {}
    for task in classification_tasks:
        targets = batch[task]
        valid = targets != -1
        if bool(valid.any()):
            losses[task] = criterion(logits_dict[task][valid], targets[valid])
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 1e-4)))

    # CrossEntropyLoss with ignore_index=-1 handles the missing labels
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
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

    best_val_loss = float("inf")
    start_epoch = 0
    optimizer_step = 0
    resume_microbatch_idx = 0
    if resume_path and Path(resume_path).exists():
        print(f"[*] Resuming from checkpoint: {resume_path}", flush=True)
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch_complete = bool(checkpoint.get("epoch_complete", True))
        start_epoch = checkpoint["epoch"] + (1 if epoch_complete else 0)
        resume_microbatch_idx = (
            0 if epoch_complete else int(checkpoint.get("microbatch_idx", 0))
        )
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        optimizer_step = int(checkpoint.get("optimizer_step", 0))
        print(
            f"[*] Resumed checkpoint. Next epoch: {start_epoch + 1}, "
            f"microbatch: {resume_microbatch_idx} "
            f"with best val loss: {best_val_loss:.4f}",
            flush=True,
        )

    print("[*] Starting Multi-Task Training...", flush=True)
    epochs = cfg.get("epochs", 5)
    grad_accum_steps = cfg.get("grad_accum_steps", 1)
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

    if not run_id:
        run_id = cfg.get("run_id", None)
    if not run_id:
        from datetime import date

        today = date.today().strftime("%Y-%m-%d")
        tag = Path(config_path).stem
        run_id = f"{today}_{tag}_{model_cfg.n_layer}L{model_cfg.n_head}H_d{model_cfg.n_embd}_e{epochs}"

    runs_dir = Path("runs") / run_id
    out_dir = runs_dir / "checkpoints"
    scores_dir = runs_dir / "scores"

    out_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)
    run_logger = RunLogger(runs_dir / "logs" / "train.log")
    run_logger.__enter__()

    log_csv = scores_dir / "curves.csv"
    import csv

    if not log_csv.exists():
        with open(log_csv, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_loss"])

    time_limit_reached = False
    start_time = time.perf_counter()
    wall_timer = WallTimer(max_time_minutes)
    checkpoint_policy = PeriodicCheckpointPolicy(
        every_steps=int(cfg.get("checkpoint_every_steps", 0) or 0),
        every_minutes=float(cfg.get("checkpoint_every_minutes", 0.0) or 0.0),
    )
    current_microbatch_idx = 0

    def checkpoint_payload(epoch_idx: int) -> dict:
        return {
            "epoch": epoch_idx,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "optimizer_step": optimizer_step,
            "microbatch_idx": current_microbatch_idx,
            "epoch_complete": False,
            "cfg": cfg,
            "dataset_provenance": dataset_provenance,
            "task_vocabs": vocabs,
            "model_spec": {
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
            },
        }

    def save_last(epoch_idx: int, reason: str) -> None:
        payload = checkpoint_payload(epoch_idx)
        payload["checkpoint_reason"] = reason
        payload["epoch_complete"] = reason == "epoch"
        if payload["epoch_complete"]:
            payload["microbatch_idx"] = 0
        save_checkpoint_atomic(payload, out_dir / "last_critic.pt")
        checkpoint_policy.mark_saved(optimizer_step)
        print(
            f"[checkpoint] saved {out_dir / 'last_critic.pt'} reason={reason} step={optimizer_step}"
        )

    for epoch in range(start_epoch, epochs):
        if time_limit_reached:
            break
        epoch_started = time.perf_counter()
        if dynamic_padding:
            train_loader.batch_sampler.set_epoch(epoch)
        model.train()
        train_loss = 0.0
        recent_loss = 0.0
        recent_steps = 0
        recent_sequences = 0
        recent_residues = 0
        recent_task_sums = {}
        recent_task_counts = {}
        recent_started = time.perf_counter()
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            if epoch == start_epoch and step < resume_microbatch_idx:
                continue
            current_microbatch_idx = step + 1
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            logits_dict = model(input_ids, attention_mask=attention_mask)

            loss = 0
            tasks_added = 0

            # Legacy motif supervision is opt-in and disabled for corrected critics.
            saliency_regularizer_weight = float(
                cfg.get("saliency_regularizer_weight", 0.0)
            )
            if saliency_regularizer_weight > 0.0 and "attention_weights" in logits_dict:
                attn_weights = logits_dict["attention_weights"]  # (B, T)
                saliency_loss = 0.0
                saliency_count = 0
                MOTIFS = ["GDSGG", "HIGH", "KMSKS", "DXD"]
                for i, seq in enumerate(batch["sequence"]):
                    active_indices = []
                    for motif in MOTIFS:
                        start_idx = seq.find(motif)
                        if start_idx != -1:
                            for offset in range(len(motif)):
                                idx = start_idx + 1 + offset
                                if idx < attn_weights.shape[1]:
                                    active_indices.append(idx)
                    if active_indices:
                        active_mass = attn_weights[i, active_indices].sum()
                        saliency_loss += -torch.log(active_mass + 1e-8)
                        saliency_count += 1
                if saliency_count > 0:
                    loss += saliency_regularizer_weight * (
                        saliency_loss / saliency_count
                    )
            supervised_losses = task_losses(
                logits_dict,
                {
                    task: batch[task].to(device)
                    for task in (*classification_tasks, *regression_tasks)
                },
                classification_tasks,
                regression_tasks,
                criterion,
            )
            if supervised_losses:
                loss += torch.stack(list(supervised_losses.values())).mean()
                tasks_added += 1
                for task, task_loss in supervised_losses.items():
                    recent_task_sums[task] = (
                        recent_task_sums.get(task, torch.zeros((), device=device))
                        + task_loss.detach()
                    )
                    recent_task_counts[task] = recent_task_counts.get(task, 0) + 1
            for task in multi_label_tasks:
                targets = batch[task].to(device)
                if targets.numel() and (targets >= 0).any():
                    loss += multi_label_criteria[task](logits_dict[task], targets)
                    tasks_added += 1

            if tasks_added > 0:
                group_size = accumulation_group_size(
                    step, len(train_loader), grad_accum_steps
                )
                loss = loss / group_size
                loss.backward()
                train_loss += loss.item() * group_size
                recent_loss += loss.item() * group_size
                recent_steps += 1
                recent_sequences += input_ids.shape[0]
                recent_residues += int(attention_mask.sum().item())

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
                optimizer_step += 1

                if checkpoint_policy.should_save(optimizer_step):
                    save_last(epoch, reason="periodic")

            if log_every_steps and (step + 1) % log_every_steps == 0:
                elapsed = time.perf_counter() - start_time
                interval_elapsed = time.perf_counter() - recent_started
                avg_recent_loss = recent_loss / max(recent_steps, 1)
                task_text = " ".join(
                    f"{task}_loss={float(total.cpu()) / recent_task_counts[task]:.4f}"
                    for task, total in sorted(recent_task_sums.items())
                )
                task_suffix = f" {task_text}" if task_text else ""
                memory_suffix = mps_memory_summary() if device.type == "mps" else ""
                print(
                    f"[progress] epoch={epoch + 1}/{epochs} "
                    f"step={step + 1}/{len(train_loader)} "
                    f"elapsed_min={elapsed / 60:.1f} "
                    f"recent_loss={avg_recent_loss:.4f} "
                    f"optimizer_step={optimizer_step} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e} "
                    f"seq_per_sec={recent_sequences / max(interval_elapsed, 1e-9):.2f} "
                    f"residues_per_sec={recent_residues / max(interval_elapsed, 1e-9):.0f} "
                    f"batch_seq_len={input_ids.shape[1]}"
                    f"{task_suffix}{memory_suffix}",
                    flush=True,
                )
                recent_loss = 0.0
                recent_steps = 0
                recent_sequences = 0
                recent_residues = 0
                recent_task_sums = {}
                recent_task_counts = {}
                recent_started = time.perf_counter()

            # Check wall-time limit at the end of every step
            if wall_timer.expired():
                print(
                    f"\n[info] Wall-time limit of {max_time_minutes} minutes reached mid-epoch.",
                    flush=True,
                )
                optimizer_boundary = (step + 1) % grad_accum_steps == 0 or (
                    step + 1
                ) == len(train_loader)
                if not optimizer_boundary:
                    optimizer.zero_grad(set_to_none=True)
                    current_microbatch_idx = (
                        step // grad_accum_steps
                    ) * grad_accum_steps
                save_last(epoch, reason="wall_time")
                print(
                    f"[success] Gracefully saved checkpoint to {out_dir / 'last_critic.pt'}. Exiting.",
                    flush=True,
                )
                time_limit_reached = True
                break

        if time_limit_reached:
            break
        resume_microbatch_idx = 0
        train_loss /= len(train_loader)
        if device.type == "mps":
            torch.mps.empty_cache()

        model.eval()
        val_loss = 0.0
        val_tasks_total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                logits_dict = model(input_ids, attention_mask=attention_mask)

                batch_loss = 0
                batch_tasks = 0
                supervised_losses = task_losses(
                    logits_dict,
                    {
                        task: batch[task].to(device)
                        for task in (*classification_tasks, *regression_tasks)
                    },
                    classification_tasks,
                    regression_tasks,
                    criterion,
                )
                if supervised_losses:
                    batch_loss += torch.stack(list(supervised_losses.values())).mean()
                    batch_tasks += 1
                for task in multi_label_tasks:
                    targets = batch[task].to(device)
                    if targets.numel() and (targets >= 0).any():
                        batch_loss += multi_label_criteria[task](
                            logits_dict[task], targets
                        )
                        batch_tasks += 1

                if batch_tasks > 0:
                    val_loss += batch_loss.item()
                    val_tasks_total += 1

        if val_tasks_total > 0:
            val_loss /= val_tasks_total
        if device.type == "mps":
            torch.mps.empty_cache()
        print(
            f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}",
            flush=True,
        )
        print(
            f"[timing] epoch={epoch + 1} wall_sec={time.perf_counter() - epoch_started:.2f}",
            flush=True,
        )

        with open(log_csv, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, f"{train_loss:.4f}", f"{val_loss:.4f}"])

        improved = False
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            improved = True

        # Save last checkpoint for resilience
        save_last(epoch, reason="epoch")

        if improved:
            best_payload = checkpoint_payload(epoch)
            best_payload["checkpoint_reason"] = "best_epoch"
            best_payload["epoch_complete"] = True
            best_payload["microbatch_idx"] = 0
            save_checkpoint_atomic(best_payload, out_dir / "best_critic.pt")
            print("  -> Saved new best model.", flush=True)


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
