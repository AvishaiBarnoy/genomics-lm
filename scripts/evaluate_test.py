#!/usr/bin/env python3
"""
Evaluate validation or test cross-entropy and perplexity for a trained run.

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
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.codonlm.checkpoints import build_codon_model_from_cfg, load_codon_checkpoint
from src.codonlm.data_loading import PackedDataset, dynamic_lm_collate_fn
from src.codonlm.evaluation_provenance import (
    artifact_provenance,
    bind_checkpoint_dataset,
    bind_dataset_manifest,
    bind_derived_dataset,
)
from src.codonlm.dataset_manifest import manifest_artifact_path
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


def _find_validation_npz(cfg: dict, repo_root: Path) -> Path:
    val_npz_cfg = cfg.get("val_npz")
    if isinstance(val_npz_cfg, list) and val_npz_cfg:
        val_npz_cfg = val_npz_cfg[0]
    if not val_npz_cfg:
        raise ValueError("validation evaluation requires val_npz in the run config")
    path = Path(val_npz_cfg)
    return path if path.is_absolute() else repo_root / path


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    device: torch.device,
    loader: DataLoader,
    *,
    label_smoothing: float = 0.0,
) -> tuple[float, float, float, int]:
    total_nll = 0.0
    total_objective = 0.0
    total_tokens = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits, _ = model(xb)
        valid = (yb != 0).sum().item()
        if valid == 0:
            continue
        flat_logits = logits.float().reshape(-1, logits.size(-1))
        flat_targets = yb.reshape(-1)
        total_nll += float(
            F.cross_entropy(
                flat_logits,
                flat_targets,
                ignore_index=0,
                reduction="sum",
            ).item()
        )
        total_objective += float(
            F.cross_entropy(
                flat_logits,
                flat_targets,
                ignore_index=0,
                reduction="sum",
                label_smoothing=float(label_smoothing),
            ).item()
        )
        total_tokens += valid
    mean_nll = total_nll / max(1, total_tokens)
    mean_objective = total_objective / max(1, total_tokens)
    ppl = float(math.exp(min(20.0, mean_nll)))
    return mean_nll, ppl, mean_objective, total_tokens


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", help="outputs/checkpoints/<RUN_ID>")
    ap.add_argument("--run_id", help="Run id (alternative to --run_dir)")
    ap.add_argument(
        "--split",
        choices=("test", "validation"),
        default="test",
        help="Frozen dataset split to evaluate (default: test).",
    )
    ap.add_argument(
        "--data_dir", help="override test NPZ directory (contains test_bs*.npz)"
    )
    ap.add_argument("--test_npz", type=Path, help="explicit test or control dataset")
    ap.add_argument(
        "--manifest",
        type=Path,
        help="Frozen dataset manifest; required for corrected checkpoints.",
    )
    ap.add_argument(
        "--derived_provenance",
        type=Path,
        help="Provenance sidecar when --test_npz is derived from the frozen test set.",
    )
    ap.add_argument(
        "--checkpoint-name",
        default="best.pt",
        help="Checkpoint filename under the run directory (default: best.pt).",
    )
    ap.add_argument(
        "--metric-prefix",
        default=None,
        help="Metrics key prefix; defaults to the selected split name.",
    )
    ap.add_argument("--batch_size", type=int, default=None, help="evaluation batch size")
    args = ap.parse_args()
    metric_prefix = args.metric_prefix or args.split
    if not metric_prefix.replace("_", "").isalnum():
        raise ValueError("--metric-prefix must contain only letters, numbers, and underscores")
    if args.split == "validation" and args.test_npz is not None:
        raise ValueError("--test_npz cannot be used with --split validation")
    if args.split == "validation" and args.derived_provenance is not None:
        raise ValueError("--derived_provenance is only supported for the test split")

    # accept run_id or run_dir
    run_id, run_dir = resolve_run(args.run_id, args.run_dir)
    repo_root = Path(__file__).resolve().parents[1]

    state_dict, cfg, checkpoint_path = load_codon_checkpoint(
        run_dir, ckpt_name=args.checkpoint_name
    )
    model = build_codon_model_from_cfg(cfg)
    model.load_state_dict(state_dict, strict=False)
    model.to(dev()).eval()

    data_dir_opt = Path(args.data_dir) if args.data_dir else None
    if args.split == "validation":
        evaluation_npz = _find_validation_npz(cfg, repo_root)
        artifact_role = "val_tokens"
    else:
        evaluation_npz = args.test_npz or _find_test_npz(
            run_id, cfg, repo_root, data_dir_opt
        )
        artifact_role = "test_tokens"
    evaluation_npz = evaluation_npz.expanduser().resolve()
    manifest_provenance = None
    derived_provenance = None
    if args.manifest is not None:
        if args.derived_provenance is None:
            _, manifest_provenance = bind_dataset_manifest(
                args.manifest, expected_artifacts={artifact_role: evaluation_npz}
            )
        else:
            manifest, manifest_provenance = bind_dataset_manifest(args.manifest)
            source_test = manifest_artifact_path(
                manifest, args.manifest.expanduser().resolve(), "test_tokens"
            )
            derived_provenance = bind_derived_dataset(
                evaluation_npz,
                args.derived_provenance,
                manifest_provenance=manifest_provenance,
                source_artifact_path=source_test,
            )
    elif args.derived_provenance is not None:
        raise ValueError("--derived_provenance requires --manifest")
    checkpoint_dataset = bind_checkpoint_dataset(cfg, manifest_provenance)
    ds = PackedDataset(evaluation_npz)

    collate_fn = dynamic_lm_collate_fn if getattr(ds, "is_dynamic", False) else None

    batch_size = args.batch_size
    if batch_size is None:
        batch_size = int(cfg.get("eval_batch_size", 16 if dev().type == "mps" else 64))
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn)
    label_smoothing = float(cfg.get("label_smoothing", 0.0))
    nll, ppl, objective_loss, evaluated_tokens = evaluate(
        model,
        dev(),
        loader,
        label_smoothing=label_smoothing,
    )
    print(
        f"[{args.split}] nll={nll:.4f} ppl={ppl:.2f} "
        f"objective={objective_loss:.4f} label_smoothing={label_smoothing:.4f}"
    )

    metrics_path = run_dir / "scores" / "metrics.json"
    prefix = metric_prefix

    write_merge_metrics(
        metrics_path,
        {
            f"{prefix}_loss": float(nll),
            f"{prefix}_nll": float(nll),
            f"{prefix}_ppl": float(ppl),
            f"{prefix}_objective_loss": float(objective_loss),
            f"{prefix}_label_smoothing": label_smoothing,
            f"{prefix}_evaluated_tokens": int(evaluated_tokens),
            f"{prefix}_evaluation_provenance": {
                "schema_version": 2,
                "loss_definition": {
                    "nll": "unsmoothed_cross_entropy",
                    "perplexity": "exp(unsmoothed_cross_entropy)",
                    "objective": "cross_entropy_with_configured_label_smoothing",
                },
                "dataset_manifest": manifest_provenance or {"status": "legacy_unverified"},
                "checkpoint_dataset": checkpoint_dataset,
                "checkpoint": artifact_provenance(checkpoint_path),
                artifact_role: artifact_provenance(evaluation_npz),
                "derived_dataset": derived_provenance,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(f"[metrics] updated {metrics_path}")


if __name__ == "__main__":
    main()
