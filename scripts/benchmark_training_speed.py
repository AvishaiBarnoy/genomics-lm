"""
scripts/benchmark_training_speed.py — Training throughput benchmark.

Measures tokens per second for a mini-run (N steps) under different
optimization configurations and prints a comparison table.

Usage:
    python -m scripts.benchmark_training_speed --config configs/stage2.6_large_scaling.yaml
    python -m scripts.benchmark_training_speed --config configs/stage2.6_optimized.yaml
    python -m scripts.benchmark_training_speed  # auto-runs both and compares
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml


def _load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _build_model(cfg: dict, device: torch.device):
    from src.codonlm.model_tiny_gpt import TinyGPT
    model = TinyGPT(
        cfg["vocab_size"],
        cfg["block_size"],
        cfg["n_layer"],
        cfg["n_head"],
        cfg["n_embd"],
        cfg.get("dropout", 0.0),
        n_kv_head=cfg.get("n_kv_head"),
        use_sdpa=cfg.get("use_sdpa", False),
    ).to(device)
    model.train()
    return model


def _build_loader(cfg: dict):
    from src.codonlm.train_codon_lm import (
        PackedDataset, MmapPackedDataset, BucketBatchSampler,
    )
    from torch.utils.data import DataLoader

    train_npz = cfg.get("train_npz", cfg.get("train_paths", []))
    if isinstance(train_npz, str):
        train_npz = [train_npz]

    use_mmap = bool(cfg.get("use_mmap", False))
    DatasetCls = MmapPackedDataset if use_mmap else PackedDataset
    ds = DatasetCls(train_npz)

    collate_fn = None
    if getattr(ds, "is_dynamic", False):
        def collate_fn(batch):
            lengths = [len(s) for s in batch]
            max_len = max(lengths)
            xs, ys = [], []
            for seq in batch:
                x = seq[:-1]; y = seq[1:]
                pad = (max_len - 1) - len(x)
                if pad > 0:
                    x = torch.cat([x, torch.zeros(pad, dtype=torch.long)])
                    y = torch.cat([y, torch.zeros(pad, dtype=torch.long)])
                xs.append(x); ys.append(y)
            return torch.stack(xs), torch.stack(ys)

    bucket_batching = bool(cfg.get("bucket_batching", False))
    batch_size = cfg["batch_size"]

    if bucket_batching and getattr(ds, "is_dynamic", False) and hasattr(ds, "seq_lengths"):
        lengths = ds.seq_lengths
        sampler = BucketBatchSampler(lengths, batch_size=batch_size, n_buckets=8, shuffle=True)
        loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate_fn)
    else:
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    return loader


def benchmark(
    config_path: str,
    n_steps: int = 30,
    warmup_steps: int = 5,
    device: Optional[torch.device] = None,
) -> dict:
    """Run N forward+backward steps and return throughput metrics."""
    cfg = _load_cfg(config_path)
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    model = _build_model(cfg, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    sep_mask_enabled = bool(cfg.get("sep_mask_enabled", True))

    try:
        loader = _build_loader(cfg)
        loader_iter = iter(loader)
    except Exception as exc:
        return {"error": str(exc), "config": config_path}

    # Metrics
    total_tokens = 0
    step_times = []

    for step in range(n_steps + warmup_steps):
        try:
            xb, yb = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            xb, yb = next(loader_iter)

        xb, yb = xb.to(device), yb.to(device)

        t0 = time.perf_counter()
        optimizer.zero_grad()

        # TinyGPT returns (logits, loss); pass targets for built-in CE loss
        logits, loss = model(xb, targets=yb)
        B, T = xb.shape
        loss.backward()
        optimizer.step()

        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()

        t1 = time.perf_counter()

        if step >= warmup_steps:
            step_times.append(t1 - t0)
            total_tokens += B * T

    avg_step_ms = np.mean(step_times) * 1000
    tokens_per_sec = total_tokens / sum(step_times)

    n_params = sum(p.numel() for p in model.parameters())

    return {
        "config": Path(config_path).stem,
        "device": str(device),
        "n_params": n_params,
        "batch_size": cfg["batch_size"],
        "n_kv_head": cfg.get("n_kv_head", "MHA"),
        "use_mmap": cfg.get("use_mmap", False),
        "bucket_batching": cfg.get("bucket_batching", False),
        "sep_mask_enabled": sep_mask_enabled,
        "use_sdpa": cfg.get("use_sdpa", False),
        "avg_step_ms": round(avg_step_ms, 1),
        "tokens_per_sec": round(tokens_per_sec),
        "n_steps": n_steps,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="Training throughput benchmark")
    ap.add_argument("--config", nargs="+", default=None, help="Config(s) to benchmark")
    ap.add_argument("--steps", type=int, default=30, help="Benchmark steps (default: 30)")
    ap.add_argument("--warmup", type=int, default=5, help="Warmup steps (default: 5)")
    args = ap.parse_args(argv)

    configs = []
    if args.config:
        configs = args.config  # already a list from nargs='+'
    else:
        # Auto-compare baseline vs optimized
        configs = [
            "configs/stage2.6_large_scaling.yaml",
            "configs/stage2.6_optimized.yaml",
        ]
        print("[bench] No --config specified. Comparing baseline vs. optimized.")

    results = []
    for cfg_path in configs:
        if not Path(cfg_path).exists():
            print(f"[bench] Skipping {cfg_path} (not found)")
            continue
        print(f"\n[bench] Running: {cfg_path} ...")
        r = benchmark(cfg_path, n_steps=args.steps, warmup_steps=args.warmup)
        if "error" in r:
            print(f"  ERROR: {r['error']}")
        else:
            print(f"  tokens/sec: {r['tokens_per_sec']:,}  |  avg step: {r['avg_step_ms']}ms")
        results.append(r)

    if len(results) >= 2 and "error" not in results[0] and "error" not in results[1]:
        speedup = results[1]["tokens_per_sec"] / results[0]["tokens_per_sec"]
        print(f"\n{'='*60}")
        print(f"SPEEDUP:  {speedup:.2f}×  ({results[1]['tokens_per_sec']:,} vs {results[0]['tokens_per_sec']:,} tok/s)")
        print(f"{'='*60}")

    print("\n--- Full Results ---")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
