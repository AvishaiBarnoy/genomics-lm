#!/usr/bin/env python3
"""
Profile a short training run to find hotspots.

Captures a Chrome trace (JSON) and a text summary sorted by CPU time.

Usage:
  python -m scripts.profile_train --config configs/tiny_mps.yaml --run_id PROFILE_RUN --steps 200
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.profiler import profile, ProfilerActivity, schedule
import yaml
import numpy as np

from src.codonlm.train_codon_lm import PackedDataset
from src.codonlm.model_tiny_gpt import TinyGPT


def dev():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def _ensure_list(v):
    if v is None:
        return []
    if isinstance(v, (str, Path)):
        return [str(v)]
    return [str(x) for x in v]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--steps", type=int, default=200, help="number of optimizer steps to record (after warmup)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config)) or {}
    device = dev()
    torch.manual_seed(cfg.get("seed", 1337))

    # Resolve NPZs (use combined manifest paths if available)
    default_train = f"data/processed/train_bs{cfg['block_size']}.npz"
    default_val = f"data/processed/val_bs{cfg['block_size']}.npz"
    train_paths = _ensure_list(cfg.get("train_npz", default_train))
    val_paths = _ensure_list(cfg.get("val_npz", default_val))

    train_ds = PackedDataset(train_paths)
    val_ds = PackedDataset(val_paths)
    train_loader = DataLoader(train_ds, batch_size=cfg.get("batch_size", 2), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.get("batch_size", 2))

    # Build model (no compile to keep trace simple)
    model = TinyGPT(
        cfg["vocab_size"], cfg["block_size"],
        n_layer=cfg["n_layer"], n_head=cfg["n_head"], n_embd=cfg["n_embd"],
        dropout=cfg.get("dropout", 0.1), use_checkpoint=bool(cfg.get("use_checkpoint", False)),
        label_smoothing=float(cfg.get("label_smoothing", 0.0)),
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 3e-4)), weight_decay=float(cfg.get("weight_decay", 0.05)))

    # Profiler output paths
    run_dir = Path("runs") / args.run_id
    out_dir = run_dir / "profile"
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "trace.json"
    txt_path = out_dir / "summary.txt"

    # Activities
    acts = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        acts.append(ProfilerActivity.CUDA)

    # Schedule: 1 step wait, 1 warmup, args.steps active
    sched = schedule(wait=1, warmup=1, active=args.steps, repeat=1)

    def on_trace_ready(p):
        p.export_chrome_trace(str(trace_path))
        # Also dump a quick table to text
        with txt_path.open("w") as fh:
            fh.write(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=200))

    gacc = int(cfg.get("grad_accum_steps", 16))

    step = 0
    model.train(True)
    with profile(activities=acts, schedule=sched, on_trace_ready=on_trace_ready, record_shapes=True, profile_memory=True, with_stack=True) as prof:
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits, loss = model(xb, yb)
            (loss / gacc).backward()
            if (step + 1) % gacc == 0:
                optim.step(); optim.zero_grad(set_to_none=True)
            step += 1
            prof.step()
            if step >= (args.steps + 10) * gacc:
                break

    print(f"[profile] wrote {trace_path} and {txt_path}")
    print("Open the trace in Chrome: chrome://tracing (Load trace) or VS Code Torch-Profiler viewer.")


if __name__ == "__main__":
    main()

