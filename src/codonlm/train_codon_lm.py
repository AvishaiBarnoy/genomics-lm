#!/usr/bin/env python3

# TODO: combine mconf and cfg variables
"""
Train a tiny codon GPT on Apple-Silicon (MPS) or CPU.

Key parameters (YAML or CLI):
- vocab_size: 69 (64 codons + 5 specials)
- block_size: 256 (M2/8GB) or 512 (M4/16GB)
- n_layer, n_head, n_embd: capacity; see configs/tiny_mps.yaml
- batch_size: micro-batch (we use grad_accum to emulate larger batches)
- grad_accum_steps: trade compute for memory
- lr: 3e-4 (cosine decay), warmup: 200 steps
- dropout: 0.1 for small models to curb overfit
- max_steps: training length
"""

import argparse, yaml, math, time, json, csv
import numpy as np, torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from tqdm import tqdm
import os
from typing import Optional, Tuple
from .model_tiny_gpt import TinyGPT, Cfg

class PackedDataset(Dataset):
    def __init__(self, npz_path):
        d = np.load(npz_path)
        self.X = torch.from_numpy(d["X"]).long()
        self.Y = torch.from_numpy(d["Y"]).long()
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.Y[i]

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def dev():
    return "mps" if torch.backends.mps.is_available() else "cpu"

class PackedDataset(Dataset):
    """
    Loads NPZ with 'X' shaped (num_sequences, seq_len).
    Returns (input, target) where target = input shifted by +1.
    """
    def __init__(self, npz_path):
        d = np.load(npz_path, allow_pickle=False)
        x = np.asarray(d["X"])                 # safe view
        x = np.ascontiguousarray(x, dtype=np.int64)
        self.X = torch.from_numpy(x)          # (N, T)
        self.T = self.X.shape[1]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        seq = self.X[i]                        # (T,)
        # use first T-1 as input, last T-1 as target
        return seq[:-1], seq[1:]               # (T-1,), (T-1,)

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


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_npz", default="data/processed/train_bs256.npz")
    ap.add_argument("--val_npz",   default="data/processed/val_bs256.npz")
    ap.add_argument("--outdir",    default="outputs/checkpoints")
    ap.add_argument("--scores_dir", default="outputs/scores")
    ap.add_argument("--run_id", default=None, help=f"Unique run id; falls back to ${RUN_ID_ENV}")
    ap.add_argument("--epochs",    type=int, default=1)
    ap.add_argument("--batch_size",type=int, default=64)
    ap.add_argument("--lr",        type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--n_layer",   type=int, default=2)     # match ckpt if resuming
    ap.add_argument("--n_head",    type=int, default=4)
    ap.add_argument("--n_embd",    type=int, default=128)
    ap.add_argument("--block_size",type=int, default=256)   # data chunks are length 256
    ap.add_argument("--dropout",   type=float, default=0.1)
    ap.add_argument("--vocab_size",type=int, default=69)
    ap.add_argument("--config", default=None, help="YAML config with model+train hparams")
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--warmup_frac", type=float, default=0.05)
    ap.add_argument("--min_lr", type=float, default=1e-5)

    # parse once to get CLI, and get the default set too for override logic
    defaults = ap.parse_args([])               # namespace with defaults
    args = ap.parse_args()                     # actual CLI
    # If a YAML config is given, merge it: YAML -> args.defaults, then CLI overrides
    if args.config is not None:
        with open(args.config, "r") as fh:
            y = yaml.safe_load(fh) or {}
        # assign YAML values where the user did not pass an explicit CLI override
        for k, v in y.items():
            if not hasattr(args, k):
                continue
            # only take YAML if CLI value equals default
            if getattr(args, k) == getattr(defaults, k):
                setattr(args, k, v)

    return args

def run_epoch(model, loader, optimizer=None, device="cpu"):
    is_train = optimizer is not None
    model.train(is_train)
    total, n = 0.0, 0
    for x, y in loader:
        # x,y come as int64 CPU tensors from the dataset; move/cast once
        x = x.to(device, non_blocking=True).long()
        y = y.to(device, non_blocking=True).long()
        logits, loss = model(x, y)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total += float(loss.detach())
        n += 1
    return total / max(n, 1)

# --- replace your if __name__ == "__main__": block with this ---
if __name__ == "__main__":
    args = parse_args()
    # Deprecation notice: prefer v2 trainer
    import sys
    print(
        "[deprecation] train_codon_lm.py (v1) is deprecated. "
        "Use src.codonlm.train_codon_lm_v2 with configs/tiny_mps_v2.yaml instead.",
        file=sys.stderr,
    )
    run_id = _normalize_run_id(args.run_id or os.environ.get(RUN_ID_ENV))
    ckpt_dir, scores_dir = _prepare_output_dirs(args.outdir, args.scores_dir, run_id)

    device = dev()
    print(f"[device] {device}")
    print(f"[cfg]   n_layer={args.n_layer} n_head={args.n_head} n_embd={args.n_embd} block_size={args.block_size} vocab={args.vocab_size}")
    print(f"[train] batch={args.batch_size} epochs={args.epochs} lr={args.lr} wd={args.weight_decay}")
    if run_id:
        print(f"[run]   id={run_id}")
    print(f"[paths] ckpts={ckpt_dir} scores={scores_dir}")

    # ---- data ----
    train_ds = PackedDataset(args.train_npz)
    val_ds   = PackedDataset(args.val_npz)

    # Inputs are length T-1; ensure model block_size can cover it
    eff_ctx = train_ds.T - 1
    if args.block_size < eff_ctx:
        print(f"[warn] Increasing block_size from {args.block_size} -> {eff_ctx} to fit context")
        args.block_size = eff_ctx

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, pin_memory=False)

    # ---- configs ----
    # Training hparams: keep them in a dict (index with ["..."])
    cfg = dict(
        lr=args.lr, weight_decay=args.weight_decay,
        epochs=args.epochs, batch_size=args.batch_size
    )
    if run_id:
        cfg["run_id"] = run_id
    # Model arch: use Cfg (attribute access)
    mconf = Cfg(
        vocab_size=args.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        block_size=args.block_size,
        dropout=args.dropout,
    )

    # ---- model/opt ----
    model = TinyGPT(mconf).to(device)

    # param groups: decay (matrices) vs no_decay (biases, layernorm)
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.ndim >= 2 else no_decay).append(p)

    optim = torch.optim.AdamW(
        [{"params": decay,    "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg["lr"],
)

    steps_per_epoch = math.ceil(len(train_loader) / max(1, args.grad_accum))
    total_steps     = steps_per_epoch * args.epochs
    warmup_steps    = max(1, int(args.warmup_frac * total_steps))

    # cosine to eta_min = args.min_lr
    base_lr = cfg["lr"]
    eta_min = args.min_lr

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * prog))  # 1 -> 0
        # scale so final LR hits eta_min
        return (eta_min / base_lr) + (1 - (eta_min / base_lr)) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    # ---- train ----
    best_val = float("inf")
    best_epoch = None
    global_step = 0
    history = []

    for epoch in range(cfg["epochs"]):
        model.train(True)
        optim.zero_grad(set_to_none=True)

        running, n = 0.0, 0
        for i, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device).long()
            y = y.to(device).long()
            _, loss = model(x, y)

            (loss / max(1, args.grad_accum)).backward()

            if i % max(1, args.grad_accum) == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                optim.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

            running += float(loss.detach())
            n += 1

        tr = running / max(1, n)
        va = run_epoch(model, val_loader, None, device)
        lr_now = scheduler.get_last_lr()[0]
        print(f"[epoch {epoch+1}] train {tr:.4f} | val {va:.4f} | lr {lr_now:.2e}")

        history.append({
            "epoch": epoch + 1,
            "train_loss": tr,
            "val_loss": va,
            "lr": lr_now,
        })

        ckpt_payload = {
            "model": model.state_dict(),
            "cfg":   dict(
                vocab_size=mconf.vocab_size, n_layer=mconf.n_layer, n_head=mconf.n_head,
                n_embd=mconf.n_embd, block_size=mconf.block_size, dropout=mconf.dropout
            ),
            "train_cfg": cfg | {"grad_accum": args.grad_accum, "warmup_frac": args.warmup_frac, "min_lr": args.min_lr},
            "epoch": epoch + 1,
            "val_loss": va,
            "train_loss": tr,
        }
        torch.save(ckpt_payload, ckpt_dir / "last.pt")

        if va < best_val:
            best_val = va
            best_epoch = epoch + 1
            torch.save(ckpt_payload, ckpt_dir / "best.pt")
            print(f"[save] {ckpt_dir / 'best.pt'} (val {va:.4f})")

    if history:
        curves_path = scores_dir / "curves.csv"
        with curves_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["epoch", "train_loss", "val_loss", "lr"])
            writer.writeheader()
            for row in history:
                writer.writerow(row)

        metrics = {
            "run_id": run_id,
            "best_epoch": best_epoch,
            "best_val_loss": best_val if best_val != float("inf") else None,
            "last_epoch": history[-1]["epoch"],
            "last_val_loss": history[-1]["val_loss"],
            "last_train_loss": history[-1]["train_loss"],
        }
        metrics_path = scores_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
