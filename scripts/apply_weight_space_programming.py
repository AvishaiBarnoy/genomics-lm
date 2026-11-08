#!/usr/bin/env python3
"""
Apply Weight-Space Programming to CodonLM checkpoint.

Directly edits neural network weights without gradient descent or training loops,
as introduced in 'Sparks of Weight-Space Programming' (Takehi et al., 2026).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
from pathlib import Path
import torch

from src.codonlm.checkpoints import build_codon_model_from_cfg, load_codon_checkpoint


def sha256_tensor(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()[:16]


def apply_weight_space_programming(
    source_run: Path | str,
    target_run: Path | str,
    head_scale: float = 1.12,
):
    source_run = Path(source_run)
    target_run = Path(target_run)
    
    # 1. Load source checkpoint
    src_ckpt_path = source_run / "checkpoints" / "best.pt"
    if not src_ckpt_path.exists():
        src_ckpt_path = source_run / "best.pt"
    
    ckpt = torch.load(src_ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model"]
    cfg = dict(ckpt.get("cfg", {}))

    # 2. Record weight diffs before edit
    orig_head = state["head.weight"].clone()
    orig_head_hash = sha256_tensor(orig_head)

    # 3. Perform weight edit: Logit calibration in weight space
    state["head.weight"] = state["head.weight"] * float(head_scale)
    new_head_hash = sha256_tensor(state["head.weight"])

    # Update metadata
    cfg["run_id"] = target_run.name
    ckpt["cfg"] = cfg
    if "model" in ckpt:
        ckpt["model"] = state

    # 4. Save to target run directory
    target_ckpt_dir = target_run / "checkpoints"
    target_ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    target_best_pt = target_ckpt_dir / "best.pt"
    target_last_pt = target_ckpt_dir / "last.pt"
    torch.save(ckpt, target_best_pt)
    shutil.copy2(target_best_pt, target_last_pt)

    # Copy / update auxiliary files
    for aux in ["itos.txt", "vocabulary.json"]:
        src_aux = source_run / aux
        if src_aux.exists():
            shutil.copy2(src_aux, target_run / aux)

    # Write target config.yaml
    cfg_yaml_path = target_ckpt_dir / "config.yaml"
    with open(cfg_yaml_path, "w") as f:
        for k, v in cfg.items():
            if v is None:
                f.write(f"{k}: null\n")
            elif isinstance(v, bool):
                f.write(f"{k}: {'true' if v else 'false'}\n")
            elif isinstance(v, (int, float, str, list)):
                f.write(f"{k}: {v}\n")

    meta = {
        "run_id": target_run.name,
        "source_run": source_run.name,
        "method": "weight_space_programming",
        "description": "Direct weight-space calibration edit inspired by Takehi et al. (2026)",
        "head_scale": head_scale,
        "edited_parameters": int(orig_head.numel()),
        "orig_head_sha256": orig_head_hash,
        "new_head_sha256": new_head_hash,
        "status": "completed",
    }
    with open(target_ckpt_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Weight-space programming complete!")
    print(f"Source: {source_run} -> Target: {target_run}")
    print(f"Edited tensor: head.weight (shape: {list(orig_head.shape)})")
    print(f"Scale: {head_scale}")
    print(f"Hash before: {orig_head_hash} -> Hash after: {new_head_hash}")
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="runs/corrected-multi-offset-heads-seed1337")
    parser.add_argument("--target", default="runs/corrected-weight-space-programming-seed1337")
    parser.add_argument("--scale", type=float, default=1.12)
    args = parser.parse_args()
    apply_weight_space_programming(args.source, args.target, args.scale)
