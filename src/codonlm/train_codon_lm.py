#!/usr/bin/env python3
"""
Training with:
- MPS autocast float16 (Apple Silicon)
- optional gradient checkpointing
- cosine or plateau LR schedules with early stopping
- CSV logging

Config highlights:
  amp: true/false
  use_checkpoint: true/false
  optimizer: "adamw" or "adafactor"
  scheduler: "cosine" (default) or "plateau"
  warmup_steps: linear warmup period
  min_lr: final LR for cosine schedule / floor for plateau
  early_stop_patience: epochs without improvement before stopping
  log_csv: where to append training curves (relative to scores_dir by default)
"""

import argparse, yaml, math, csv, time, torch, numpy as np, json, os
from pathlib import Path
from typing import Optional, Tuple
from torch.utils.data import DataLoader, Dataset
from .model_tiny_gpt import TinyGPT

# optional Adafactor
try:
    from transformers.optimization import Adafactor
    HAS_ADAFACTOR = True
except Exception:
    HAS_ADAFACTOR = False

RUN_ID_ENV = "RUN_ID"


def _normalize_run_id(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    run_id = str(value).strip()
    return run_id or None


def _auto_run_id(cfg: dict, config_path: Optional[str]) -> str:
    try:
        from . import __package__  # noqa: F401
    except Exception:
        pass
    # Build RUN_ID like YYYY-MM-DD_<model>_<nL><nH>_d<n_embd>_e<epochs>
    from datetime import date
    from pathlib import Path
    today = date.today().strftime("%Y-%m-%d")
    tag = "run"
    if config_path:
        stem = Path(config_path).stem
        tag = stem.split("_", 1)[0] if "_" in stem else stem
    return f"{today}_{tag}_{int(cfg.get('n_layer',0))}L{int(cfg.get('n_head',0))}H_d{int(cfg.get('n_embd',0))}_e{int(cfg.get('epochs',0))}"


def _prepare_output_dirs(base_ckpt_dir: str, base_scores_dir: str, run_id: Optional[str]) -> Tuple[Path, Path]:
    ckpt_root = Path(base_ckpt_dir)
    if run_id:
        ckpt_root = ckpt_root / run_id
    ckpt_root.mkdir(parents=True, exist_ok=True)

    scores_root = Path(base_scores_dir)
    if run_id:
        scores_root = scores_root / run_id
    scores_root.mkdir(parents=True, exist_ok=True)
    return ckpt_root, scores_root


class PackedDataset(Dataset):
    def __init__(self, npz_path):
        d = np.load(npz_path)
        self.X = torch.from_numpy(d["X"]).long()
        self.Y = torch.from_numpy(d["Y"]).long()
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.Y[i]

def dev():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run_id", default=None, help=f"Unique run id; falls back to ${RUN_ID_ENV} or config.run_id")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    device = dev()
    torch.manual_seed(cfg.get("seed", 1337))
    amp = bool(cfg.get("amp", True)) and (device.type == "mps")

    train_npz = f"data/processed/train_bs{cfg['block_size']}.npz"
    val_npz   = f"data/processed/val_bs{cfg['block_size']}.npz"
    train_ds, val_ds = PackedDataset(train_npz), PackedDataset(val_npz)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg["batch_size"])

    model = TinyGPT(
        cfg["vocab_size"],
        cfg["block_size"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        n_embd=cfg["n_embd"],
        dropout=cfg["dropout"],
        use_checkpoint=bool(cfg.get("use_checkpoint", False)),
        label_smoothing=float(cfg.get("label_smoothing", 0.0)),
    ).to(device)

    # Optimizer selection
    if cfg.get("optimizer", "adamw").lower() == "adafactor":
        if not HAS_ADAFACTOR:
            raise RuntimeError("transformers not installed; pip install transformers to use Adafactor")
        optim = Adafactor(model.parameters(), lr=cfg.get("lr", 3e-4),
                          scale_parameter=False, relative_step=False, weight_decay=cfg.get("weight_decay", 0.05))
    else:
        optim = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    scheduler_name = str(cfg.get("scheduler", "cosine")).lower()
    if scheduler_name not in {"cosine", "plateau"}:
        print(f"[warn] Unknown scheduler '{scheduler_name}', defaulting to cosine.")
        scheduler_name = "cosine"

    gacc = cfg.get("grad_accum_steps", 16)
    warmup_steps = int(cfg.get("warmup_steps", 200))
    min_lr = float(cfg.get("min_lr", 1e-5))
    base_lr = float(cfg["lr"])

    max_epochs = int(cfg.get("epochs", 5))
    use_cosine = scheduler_name == "cosine"
    if use_cosine:
        steps_per_epoch = math.ceil(len(train_loader) / max(1, gacc))
        total_steps = max(1, steps_per_epoch * max_epochs)
        warmup_for_lambda = max(1, warmup_steps)
        min_lr_ratio = (min_lr / base_lr) if base_lr > 0 else 0.0

        def lr_lambda(step_idx: int) -> float:
            if step_idx < warmup_for_lambda:
                return float(step_idx + 1) / warmup_for_lambda
            progress = (step_idx - warmup_for_lambda) / max(1, total_steps - warmup_for_lambda)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1 - min_lr_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim,
            mode="min",
            factor=0.5,
            patience=cfg.get("plateau_patience", 2),
            min_lr=min_lr,
        )

    run_id = _normalize_run_id(args.run_id or cfg.get("run_id") or os.environ.get(RUN_ID_ENV))
    if not run_id:
        run_id = _auto_run_id(cfg, args.config)
    if run_id:
        cfg["run_id"] = run_id
    outdir = cfg["out_dir"]
    scores_base = cfg.get("scores_dir", "outputs/scores")
    ckpt_dir, scores_dir = _prepare_output_dirs(outdir, scores_base, run_id)
    log_csv_cfg = cfg.get("log_csv")
    if log_csv_cfg:
        log_csv_path = Path(log_csv_cfg)
        log_csv = (scores_dir / log_csv_path).resolve() if not log_csv_path.is_absolute() else log_csv_path
    else:
        log_csv = scores_dir / "curves.csv"
    log_csv.parent.mkdir(parents=True, exist_ok=True)
    with log_csv.open("w", newline="") as f:
        writer = csv.writer(f); writer.writerow(["step","train_loss","val_loss","perplexity","lr"])

    best = float("inf"); no_improve = 0
    step = 0

    scaler = None  # GradScaler is CUDA-only; autocast on MPS may be unsupported in some torch versions

    def one_pass(split, loader):
        nonlocal step
        mps_autocast_ok = True
        model.train(split=="train")
        total, n = 0.0, 0
        optim.zero_grad(set_to_none=True)
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            def fwd():
                logits_, loss_ = model(xb, yb)
                return logits_, loss_ / gacc

            if amp and mps_autocast_ok:
                try:
                    with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=True):
                        logits, loss = fwd()
                except RuntimeError as e:
                    msg = str(e).lower()
                    if "unsupported autocast device_type" in msg or "autocast" in msg and device.type == "mps":
                        # Fallback: disable MPS autocast for this run
                        mps_autocast_ok = False
                        logits, loss = fwd()
                    else:
                        raise
            else:
                logits, loss = fwd()
            if split=="train":
                loss.backward()
                if (n+1) % gacc == 0:
                    if (not use_cosine) and warmup_steps > 0 and step < warmup_steps:
                        scale = float(step + 1) / max(1, warmup_steps)
                        for pg in optim.param_groups:
                            pg["lr"] = base_lr * scale
                    optim.step()
                    optim.zero_grad(set_to_none=True)
                    step += 1
                    if use_cosine:
                        scheduler.step()
            total += loss.item()*gacc; n += 1
        return total / max(n,1)

    history = []
    best_epoch = None
    if run_id:
        print(f"[run] id={run_id}")
    print(f"[paths] ckpts={ckpt_dir} scores={scores_dir} log_csv={log_csv}")

    for epoch in range(max_epochs):
        epoch_idx = epoch + 1
        train_loss = one_pass("train", train_loader)
        with torch.no_grad():
            val_loss = one_pass("val", val_loader)
        ppl = math.exp(min(20.0, val_loss))
        if not use_cosine:
            scheduler.step(val_loss)
        lr_now = optim.param_groups[0]["lr"]
        print(f"[epoch {epoch_idx}] train {train_loss:.3f} | val {val_loss:.3f} | ppl {ppl:.2f} | lr {lr_now:.2e}")

        ckpt_payload = {
            "model": model.state_dict(),
            "cfg": cfg,
            "epoch": epoch_idx,
            "val_loss": val_loss,
            "train_loss": train_loss,
        }
        torch.save(ckpt_payload, ckpt_dir / "last.pt")
        with log_csv.open("a", newline="") as f:
            csv.writer(f).writerow([epoch_idx, f"{train_loss:.4f}", f"{val_loss:.4f}", f"{ppl:.3f}", f"{lr_now:.3e}"])

        history.append({
            "epoch": epoch_idx,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "perplexity": ppl,
            "lr": lr_now,
        })

        if val_loss + 1e-6 < best:
            best = val_loss; no_improve = 0
            best_epoch = epoch_idx
            torch.save(ckpt_payload, ckpt_dir / "best.pt")
        else:
            no_improve += 1
            if no_improve >= int(cfg.get("early_stop_patience", 5)):
                print("[early-stopping] no improvement; stopping.")
                break

    if history:
        metrics = {
            "run_id": run_id,
            "best_epoch": best_epoch,
            "best_val_loss": best if best != float("inf") else None,
            "last_epoch": history[-1]["epoch"],
            "last_val_loss": history[-1]["val_loss"],
            "last_train_loss": history[-1]["train_loss"],
            "last_perplexity": history[-1]["perplexity"],
        }
        metrics_path = scores_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

if __name__ == "__main__":
    main()
