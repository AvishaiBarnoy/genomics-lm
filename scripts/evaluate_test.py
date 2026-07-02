#!/usr/bin/env python3
"""
Evaluate test cross-entropy and perplexity for a trained run.

Usage:
  python -m scripts.evaluate_test --run_dir outputs/checkpoints/<RUN_ID>
  python -m scripts.evaluate_test --run_dir outputs/checkpoints/<RUN_ID> --data_dir data/processed/combined/<RUN_ID>
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from src.codonlm.checkpoints import build_codon_model_from_cfg, load_codon_checkpoint
from src.codonlm.data_loading import PackedDataset, dynamic_lm_collate_fn
from scripts._shared import resolve_run
from src.codonlm.metrics_io import write_merge_metrics


def dev() -> torch.device:
    import os
    if os.environ.get("FORCE_CPU") == "1":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _infer_run_id(run_dir: Path) -> str:
    return run_dir.name


def _find_test_npz(
    run_id: str, cfg: dict, repo_root: Path, data_dir_opt: Optional[Path]
) -> Path:
    if data_dir_opt is not None:
        return data_dir_opt / f"test_bs{cfg['block_size']}.npz"
    # prefer test_npz from config
    test_npz_cfg = cfg.get("test_npz")
    if test_npz_cfg:
        if isinstance(test_npz_cfg, list) and len(test_npz_cfg) > 0:
            test_npz_cfg = test_npz_cfg[0]
        p = Path(test_npz_cfg)
        if p.is_absolute():
            return p
        return repo_root / p
    # prefer combined manifest under data/processed/combined/<RUN_ID>
    manifest = repo_root / "data/processed/combined" / run_id / "manifest.json"
    if manifest.exists():
        data = json.loads(manifest.read_text())
        t = Path(data.get("test", ""))
        return t if t.is_absolute() else (repo_root / t)
    # fallback to default layout
    return repo_root / f"data/processed/test_bs{cfg['block_size']}.npz"


@torch.no_grad()
def evaluate(
    model: torch.nn.Module, device: torch.device, loader: DataLoader
) -> tuple[float, float]:
    total_loss = 0.0
    total_tokens = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits, loss = model(xb, yb)
        if loss is None:
            continue
        # reconstruct valid token count (ignore_index=0)
        valid = (yb != 0).sum().item()
        total_loss += float(loss.item()) * max(1, valid)
        total_tokens += max(1, valid)
    mean_nll = total_loss / max(1, total_tokens)
    ppl = float(math.exp(min(20.0, mean_nll)))
    return mean_nll, ppl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", help="outputs/checkpoints/<RUN_ID>")
    ap.add_argument("--run_id", help="Run id (alternative to --run_dir)")
    ap.add_argument(
        "--data_dir", help="override test NPZ directory (contains test_bs*.npz)"
    )
    ap.add_argument("--batch_size", type=int, default=None, help="evaluation batch size")
    args = ap.parse_args()

    # accept run_id or run_dir
    run_id, run_dir = resolve_run(args.run_id, args.run_dir)
    repo_root = Path(__file__).resolve().parents[1]

    state_dict, cfg, _ = load_codon_checkpoint(run_dir)
    model = build_codon_model_from_cfg(cfg)
    model.load_state_dict(state_dict, strict=False)
    model.to(dev()).eval()

    data_dir_opt = Path(args.data_dir) if args.data_dir else None
    test_npz = _find_test_npz(run_id, cfg, repo_root, data_dir_opt)
    ds = PackedDataset(test_npz)

    collate_fn = dynamic_lm_collate_fn if getattr(ds, "is_dynamic", False) else None

    batch_size = args.batch_size
    if batch_size is None:
        batch_size = int(cfg.get("eval_batch_size", 16 if dev().type == "mps" else 64))
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn)
    nll, ppl = evaluate(model, dev(), loader)
    print(f"[test] loss={nll:.4f} ppl={ppl:.2f}")

    metrics_path = run_dir / "scores" / "metrics.json"
    if not metrics_path.parent.exists():
        metrics_path = repo_root / "outputs/scores" / run_id / "metrics.json"

    write_merge_metrics(
        metrics_path,
        {
            "test_loss": float(nll),
            "test_ppl": float(ppl),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(f"[metrics] updated {metrics_path}")


if __name__ == "__main__":
    main()
