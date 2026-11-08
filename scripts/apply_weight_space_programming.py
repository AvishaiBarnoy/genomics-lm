#!/usr/bin/env python3
"""
Apply Weight-Space Programming to CodonLM checkpoints.

Supports both conservative and aggressive tensor surgery without gradient descent:
1. Logit Calibration: global scaling of head.weight
2. Termination Subspace Steering: projecting stop codon vectors along ln_f.bias
3. Repetition Circuit Surgery: spectral dampening of runaway induction heads (L8/L9 Head 1)
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
    head_scale: float = 1.10,
    stop_steering_alpha: float = 0.010,
    repetition_scale_l9: float = 0.85,
    repetition_scale_l8: float = 0.70,
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

    # Read vocabulary
    itos_path = source_run / "itos.txt"
    if not itos_path.exists():
        itos_path = Path("data/processed/corrected/corrected-codonlm-v1/genome/itos.txt")
    with open(itos_path) as f:
        itos = [line.strip() for line in f]
    stoi = {tok: i for i, tok in enumerate(itos)}

    stop_ids = [stoi[s] for s in ["TAA", "TAG", "TGA", "<EOS_CDS>"] if s in stoi]

    orig_head_hash = sha256_tensor(state["head.weight"])
    edited_tensors = []

    # 2. Calibration: scale head.weight
    if head_scale != 1.0:
        state["head.weight"] = state["head.weight"] * float(head_scale)
        edited_tensors.append("head.weight (calibration)")

    # 3. Termination Subspace Steering along ln_f.bias
    if stop_steering_alpha > 0.0 and "ln_f.bias" in state:
        ln_f_b = state["ln_f.bias"].detach()
        b_dir = ln_f_b / torch.dot(ln_f_b, ln_f_b)
        for sid in stop_ids:
            state["head.weight"][sid] += float(stop_steering_alpha) * b_dir
        edited_tensors.append("head.weight (stop_steering)")

    # 4. Repetition Circuit Surgery on Head 1 (dim slice 48:96)
    head_dim = cfg.get("n_embd", 384) // cfg.get("n_head", 8)
    h1_slice = slice(head_dim * 1, head_dim * 2)

    if repetition_scale_l9 != 1.0 and "blocks.9.attn.proj.weight" in state:
        state["blocks.9.attn.proj.weight"][:, h1_slice] *= float(repetition_scale_l9)
        edited_tensors.append("blocks.9.attn.proj.weight (Head 1)")

    if repetition_scale_l8 != 1.0 and "blocks.8.attn.proj.weight" in state:
        state["blocks.8.attn.proj.weight"][:, h1_slice] *= float(repetition_scale_l8)
        edited_tensors.append("blocks.8.attn.proj.weight (Head 1)")

    new_head_hash = sha256_tensor(state["head.weight"])

    # Update metadata
    cfg["run_id"] = target_run.name
    ckpt["cfg"] = cfg
    ckpt["model"] = state

    # 5. Save target checkpoint
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
        "method": "weight_space_programming_aggressive",
        "description": "Aggressive weight-space programming: calibration + termination steering + induction deflation",
        "head_scale": head_scale,
        "stop_steering_alpha": stop_steering_alpha,
        "repetition_scale_l9": repetition_scale_l9,
        "repetition_scale_l8": repetition_scale_l8,
        "edited_tensors": edited_tensors,
        "orig_head_sha256": orig_head_hash,
        "new_head_sha256": new_head_hash,
        "status": "completed",
    }
    with open(target_ckpt_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Weight-space programming complete!")
    print(f"Source: {source_run} -> Target: {target_run}")
    print(f"Edited components: {edited_tensors}")
    print(f"Parameters: scale={head_scale}, alpha={stop_steering_alpha}, l9={repetition_scale_l9}, l8={repetition_scale_l8}")
    print(f"Head hash: {orig_head_hash} -> {new_head_hash}")
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="runs/corrected-codonlm-v1-batch64-lr-ablation-lr_1_5e4")
    parser.add_argument("--target", default="runs/corrected-aggressive-weight-space-programming-seed1337")
    parser.add_argument("--head_scale", type=float, default=1.10)
    parser.add_argument("--stop_steering_alpha", type=float, default=0.010)
    parser.add_argument("--repetition_scale_l9", type=float, default=0.85)
    parser.add_argument("--repetition_scale_l8", type=float, default=0.70)
    args = parser.parse_args()
    apply_weight_space_programming(
        args.source,
        args.target,
        head_scale=args.head_scale,
        stop_steering_alpha=args.stop_steering_alpha,
        repetition_scale_l9=args.repetition_scale_l9,
        repetition_scale_l8=args.repetition_scale_l8,
    )
