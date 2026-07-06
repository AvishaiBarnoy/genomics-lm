#!/usr/bin/env python3
"""
Model Checkpoint Expansion Utility.
Upscales pre-trained checkpoints to higher layer, head, or embedding capacities.
"""

from __future__ import annotations

import argparse
import os
import yaml
import torch
from pathlib import Path

from src.codonlm.model_tiny_gpt import TinyGPT

def parse_args():
    parser = argparse.ArgumentParser(description="Expand model checkpoints across different shapes.")
    parser.add_argument("--src_checkpoint", required=True, help="Path to source checkpoint .pt file")
    parser.add_argument("--dst_config", required=True, help="Path to target config YAML file")
    parser.add_argument("--out_checkpoint", required=True, help="Path to output expanded checkpoint .pt file")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"[expand] Loading source checkpoint from: {args.src_checkpoint}")
    src_ckpt = torch.load(args.src_checkpoint, map_location="cpu")
    
    if isinstance(src_ckpt, dict) and "model" in src_ckpt:
        src_state = src_ckpt["model"]
        src_cfg = src_ckpt.get("cfg", {})
    else:
        src_state = src_ckpt
        src_cfg = {}

    print(f"[expand] Loading target config from: {args.dst_config}")
    with open(args.dst_config) as f:
        dst_cfg = yaml.safe_load(f) or {}

    # Set parameters needed to build target model
    vocab_size = dst_cfg.get("vocab_size", src_cfg.get("vocab_size", 69))
    block_size = dst_cfg.get("block_size", src_cfg.get("block_size", 512))
    n_layer = dst_cfg.get("n_layer", 3)
    n_head = dst_cfg.get("n_head", 4)
    n_embd = dst_cfg.get("n_embd", 256)
    dropout = dst_cfg.get("dropout", 0.1)
    use_checkpoint = bool(dst_cfg.get("use_checkpoint", False))
    label_smoothing = float(dst_cfg.get("label_smoothing", 0.0))
    sep_mask_enabled = bool(dst_cfg.get("sep_mask_enabled", True))
    tie_embeddings = bool(dst_cfg.get("tie_embeddings", True))
    n_kv_head = dst_cfg.get("n_kv_head")
    use_sdpa = bool(dst_cfg.get("use_sdpa", False))
    use_swiglu = bool(dst_cfg.get("use_swiglu", False))
    use_rope = bool(dst_cfg.get("use_rope", False))

    # Detect multi-offset targets and termination heads
    termination_aux = bool(dst_cfg.get("termination_loss_enabled", False) or dst_cfg.get("replay_loss_enabled", False))
    termination_bucket_edges = tuple(int(x) for x in dst_cfg.get("termination_bucket_edges", [0, 3, 10, 30]))
    termination_n_classes = int(dst_cfg.get("termination_n_classes", len(termination_bucket_edges) + 1))
    multi_offset_targets = dst_cfg.get("multi_offset_targets", [])

    print(f"[expand] Creating target TinyGPT model with:")
    print(f"  n_layer={n_layer}, n_head={n_head}, n_embd={n_embd}, use_swiglu={use_swiglu}, use_rope={use_rope}")
    target_model = TinyGPT(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
        use_checkpoint=use_checkpoint,
        label_smoothing=label_smoothing,
        sep_id=(3 if sep_mask_enabled else None),
        tie_embeddings=tie_embeddings,
        n_kv_head=n_kv_head,
        use_sdpa=use_sdpa,
        termination_aux=termination_aux,
        termination_n_classes=termination_n_classes,
        multi_offset_targets=multi_offset_targets,
        use_swiglu=use_swiglu,
        use_rope=use_rope,
    )

    target_state = target_model.state_dict()
    new_state = {}
    copied_keys = 0
    expanded_keys = 0
    missing_keys = 0

    for name, target_tensor in target_state.items():
        if name in src_state:
            source_tensor = src_state[name]
            if source_tensor.shape == target_tensor.shape:
                new_state[name] = source_tensor
                copied_keys += 1
            else:
                # Shape expansion mapping
                expanded_keys += 1
                new_tensor = target_tensor.clone()
                if source_tensor.ndim == 1:
                    n_copy = min(source_tensor.shape[0], target_tensor.shape[0])
                    new_tensor[:n_copy] = source_tensor[:n_copy]
                elif source_tensor.ndim == 2:
                    out_copy = min(source_tensor.shape[0], target_tensor.shape[0])
                    in_copy = min(source_tensor.shape[1], target_tensor.shape[1])
                    new_tensor[:out_copy, :in_copy] = source_tensor[:out_copy, :in_copy]
                new_state[name] = new_tensor
        else:
            new_state[name] = target_tensor
            missing_keys += 1

    # Apply expansion
    target_model.load_state_dict(new_state)
    print(f"[expand] Completed parameter mapping: copied={copied_keys}, expanded={expanded_keys}, missing_initialized={missing_keys}")

    # Build output payload
    out_payload = {
        "model": new_state,
        "cfg": dst_cfg,
        "epoch": 0,
        "step": 0,
        "best_val": float("inf"),
        "no_improve": 0,
    }

    out_path = Path(args.out_checkpoint)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_payload, out_path)
    print(f"[success] Wrote expanded checkpoint to: {out_path}")


if __name__ == "__main__":
    main()
