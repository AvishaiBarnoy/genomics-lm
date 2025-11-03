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
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.codonlm.model_tiny_gpt import TinyGPT
from src.codonlm.metrics_io import write_merge_metrics


def dev() -> torch.device:
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def _infer_run_id(run_dir: Path) -> str:
    return run_dir.name


def _load_checkpoint(run_dir: Path) -> tuple[dict, dict]:
    best = run_dir / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"best.pt not found in {run_dir}")
    state = torch.load(best, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        return state["model"], state.get("cfg", {})
    return state, {}


def _build_model_from_cfg(cfg: dict) -> TinyGPT:
    required = ["vocab_size", "block_size", "n_layer", "n_head", "n_embd"]
    miss = [k for k in required if k not in cfg]
    if miss:
        raise RuntimeError(f"Checkpoint config missing fields: {miss}")
    return TinyGPT(
        vocab_size=int(cfg["vocab_size"]),
        block_size=int(cfg["block_size"]),
        n_layer=int(cfg["n_layer"]),
        n_head=int(cfg["n_head"]),
        n_embd=int(cfg["n_embd"]),
        dropout=float(cfg.get("dropout", 0.0)),
        use_checkpoint=False,
        label_smoothing=float(cfg.get("label_smoothing", 0.0)),
    )


def _find_test_npz(run_id: str, cfg: dict, repo_root: Path, data_dir_opt: Optional[Path]) -> Path:
    if data_dir_opt is not None:
        return data_dir_opt / f"test_bs{cfg['block_size']}.npz"
    # prefer combined manifest under data/processed/combined/<RUN_ID>
    manifest = repo_root / "data/processed/combined" / run_id / "manifest.json"
    if manifest.exists():
        data = json.loads(manifest.read_text())
        t = Path(data.get("test", ""))
        return t if t.is_absolute() else (repo_root / t)
    # fallback to default layout
    return repo_root / f"data/processed/test_bs{cfg['block_size']}.npz"


class PackedDataset(torch.utils.data.Dataset):
    def __init__(self, npz_path: Path):
        blob = np.load(npz_path)
        self.X = torch.from_numpy(np.asarray(blob["X"]).astype(np.int64))
        self.Y = torch.from_numpy(np.asarray(blob["Y"]).astype(np.int64))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.Y[i]


@torch.no_grad()
def evaluate(model: TinyGPT, device: torch.device, loader: DataLoader) -> tuple[float, float]:
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
    ppl = float(np.exp(min(20.0, mean_nll)))
    return mean_nll, ppl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="outputs/checkpoints/<RUN_ID>")
    ap.add_argument("--data_dir", help="override test NPZ directory (contains test_bs*.npz)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
    repo_root = Path(__file__).resolve().parents[1]
    run_id = _infer_run_id(run_dir)

    state_dict, cfg = _load_checkpoint(run_dir)
    model = _build_model_from_cfg(cfg)
    model.load_state_dict(state_dict, strict=False)
    model.to(dev()).eval()

    data_dir_opt = Path(args.data_dir) if args.data_dir else None
    test_npz = _find_test_npz(run_id, cfg, repo_root, data_dir_opt)
    ds = PackedDataset(test_npz)
    loader = DataLoader(ds, batch_size=int(cfg.get("batch_size", 2)))
    nll, ppl = evaluate(model, dev(), loader)
    print(f"[test] loss={nll:.4f} ppl={ppl:.2f}")

    metrics_path = repo_root / "outputs/scores" / run_id / "metrics.json"
    write_merge_metrics(metrics_path, {
        "test_loss": float(nll),
        "test_ppl": float(ppl),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    print(f"[metrics] updated {metrics_path}")


if __name__ == "__main__":
    main()

