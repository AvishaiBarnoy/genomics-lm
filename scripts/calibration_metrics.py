#!/usr/bin/env python3
"""
Compute calibration metrics (ECE, Brier) on a split using a trained checkpoint.

CLI examples:
  python -m scripts.calibration_metrics --ckpt outputs/checkpoints/<RUN_ID>/best.pt \
    --npz data/processed/combined/<RUN_ID>/val_bs512.npz --out outputs/scores/<RUN_ID>/metrics.json

Notes:
  - Ignores PAD targets (0).
  - ECE uses 15 bins on max-probability predictions.
  - Brier (multiclass) computed as mean sum_k (p_k - y_k)^2.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from src.codonlm.model_tiny_gpt import TinyGPT


def load_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as blob:
        return np.asarray(blob["X"], dtype=np.int64), np.asarray(blob["Y"], dtype=np.int64)


def ece_from_logits(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = 0, n_bins: int = 15) -> float:
    # logits: (N, V), targets: (N,)
    mask = targets.ne(ignore_index)
    if mask.sum() == 0:
        return float("nan")
    logits = logits[mask]
    targets = targets[mask]
    probs = torch.softmax(logits, dim=-1)
    conf, pred = probs.max(dim=-1)  # (N,)
    correct = pred.eq(targets).float()
    bins = torch.linspace(0, 1, steps=n_bins+1, device=probs.device)
    ece = torch.zeros((), device=probs.device)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        sel = (conf > lo) & (conf <= hi)
        n = sel.sum().item()
        if n == 0:
            continue
        acc = correct[sel].mean()
        avg_conf = conf[sel].mean()
        ece = ece + (n / mask.sum()) * (avg_conf - acc).abs()
    return float(ece.item())


def brier_from_logits(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = 0) -> float:
    mask = targets.ne(ignore_index)
    if mask.sum() == 0:
        return float("nan")
    logits = logits[mask]
    targets = targets[mask]
    probs = torch.softmax(logits, dim=-1)
    # One-hot true labels
    V = probs.size(-1)
    y = torch.zeros_like(probs)
    y[torch.arange(y.size(0), device=y.device), targets] = 1.0
    brier = ((probs - y) ** 2).sum(dim=-1).mean()
    return float(brier.item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--npz", required=True, help="NPZ with X,Y (val or test split)")
    ap.add_argument("--out", required=True, help="metrics.json to merge into")
    args = ap.parse_args()

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    # Load checkpoint
    state = torch.load(args.ckpt, map_location="cpu")
    sd = state.get("model", state)
    cfg = state.get("cfg", {})
    V = int(cfg.get("vocab_size", 68))
    model = TinyGPT(
        vocab_size=V,
        block_size=int(cfg.get("block_size", 256)),
        n_layer=int(cfg.get("n_layer", 4)),
        n_head=int(cfg.get("n_head", 4)),
        n_embd=int(cfg.get("n_embd", 256)),
        dropout=float(cfg.get("dropout", 0.0)),
        use_checkpoint=False,
        label_smoothing=float(cfg.get("label_smoothing", 0.0)),
        sep_id=(3 if bool(cfg.get("sep_mask_enabled", True)) else None),
    )
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()

    # Load split
    X, Y = load_npz(Path(args.npz))  # (N, T)
    batch = 32
    ece_accum, brier_accum, count = 0.0, 0.0, 0
    with torch.no_grad():
        for i in range(0, X.shape[0], batch):
            xb = torch.from_numpy(X[i:i+batch]).to(device)
            yb = torch.from_numpy(Y[i:i+batch]).to(device)
            logits, _ = model(xb, yb)
            # Flatten per-token
            logits_flat = logits.reshape(-1, logits.size(-1))
            targets_flat = yb.reshape(-1)
            ece = ece_from_logits(logits_flat, targets_flat, ignore_index=0)
            brier = brier_from_logits(logits_flat, targets_flat, ignore_index=0)
            n_tok = targets_flat.ne(0).sum().item()
            # Weighted average by token count
            ece_accum += ece * n_tok
            brier_accum += brier * n_tok
            count += n_tok
    metrics = {}
    if count > 0:
        metrics["ece_val"] = float(ece_accum / count)
        metrics["brier_val"] = float(brier_accum / count)

    outp = Path(args.out)
    try:
        prev = json.loads(outp.read_text()) if outp.exists() else {}
    except Exception:
        prev = {}
    prev.update(metrics)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(prev, indent=2, sort_keys=True))
    print(f"[calibration] merged {metrics} into {outp}")


if __name__ == "__main__":
    main()

