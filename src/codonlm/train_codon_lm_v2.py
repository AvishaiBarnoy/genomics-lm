#!/usr/bin/env python3
"""
Training with:
- MPS autocast float16 (Apple Silicon)
- optional gradient checkpointing
- ReduceLROnPlateau scheduler + early stopping
- CSV logging

New config keys:
  amp: true/false
  use_checkpoint: true/false
  early_stop_patience: 5
  log_csv: "outputs/train_log.csv"
  optimizer: "adamw" or "adafactor"
"""

import argparse, yaml, math, csv, time, torch, numpy as np, json, os
from pathlib import Path
from typing import Optional, Tuple
from torch.utils.data import DataLoader, Dataset
from .model_tiny_gpt_v2 import TinyGPTv2

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

    model = TinyGPTv2(cfg["vocab_size"], cfg["block_size"],
                      n_layer=cfg["n_layer"], n_head=cfg["n_head"], n_embd=cfg["n_embd"],
                      dropout=cfg["dropout"], use_checkpoint=bool(cfg.get("use_checkpoint", False))).to(device)

    # Optimizer selection
    if cfg.get("optimizer", "adamw").lower() == "adafactor":
        if not HAS_ADAFACTOR:
            raise RuntimeError("transformers not installed; pip install transformers to use Adafactor")
        optim = Adafactor(model.parameters(), lr=cfg.get("lr", 3e-4),
                          scale_parameter=False, relative_step=False, weight_decay=cfg.get("weight_decay", 0.05))
    else:
        optim = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", factor=0.5,
                                                           patience=cfg.get("plateau_patience", 2), min_lr=1e-6)
    warmup = cfg.get("warmup_steps", 200)
    gacc = cfg.get("grad_accum_steps", 16)

    run_id = _normalize_run_id(args.run_id or cfg.get("run_id") or os.environ.get(RUN_ID_ENV))
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

    scaler = None  # GradScaler is CUDA-only; autocast on MPS works without scaler

    def one_pass(split, loader):
        nonlocal step
        model.train(split=="train")
        total, n = 0.0, 0
        optim.zero_grad(set_to_none=True)
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            with torch.amp.autocast(device_type="mps", dtype=torch.float16, enabled=amp):
                logits, loss = model(xb, yb)
                loss = loss / gacc
            if split=="train":
                loss.backward()
                if (n+1) % gacc == 0:
                    # warmup (linear)
                    if step < warmup:
                        for pg in optim.param_groups:
                            pg["lr"] = cfg["lr"] * float(step+1)/warmup
                    optim.step(); optim.zero_grad(set_to_none=True)
                    step += 1
            total += loss.item()*gacc; n += 1
        return total / max(n,1)

    max_epochs = int(cfg.get("epochs", 5))
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
