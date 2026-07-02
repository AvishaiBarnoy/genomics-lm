#!/usr/bin/env python3
"""
Compute validation perplexity for a saved checkpoint on the packed val set.
"""

import argparse
import torch
from .model_tiny_gpt import TinyGPT
from .data_loading import PackedDataset

def dev():
    """Returns the torch device to use, preferring mps (metal performance shaders) if available."""
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

def main():
    """Computes validation perplexity for a saved checkpoint on the val set."""

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val_npz", required=True)
    args = ap.parse_args()

    # 1) load checkpoint
    state = torch.load(args.ckpt, map_location="cpu")
    sd = state.get("model", state)
    cfg_saved = state.get("cfg", None)

    cfg_src = cfg_saved or {
        "vocab_size": getattr(args, "vocab_size", 69),
        "block_size": getattr(args, "block_size", 256),
        "n_layer": getattr(args, "n_layer", 2),
        "n_head": getattr(args, "n_head", 4),
        "n_embd": getattr(args, "n_embd", 128),
        "dropout": getattr(args, "dropout", 0.0),
        "use_checkpoint": False,
        "label_smoothing": getattr(args, "label_smoothing", 0.0),
    }
    model = TinyGPT(
        vocab_size=cfg_src["vocab_size"],
        block_size=cfg_src["block_size"],
        n_layer=cfg_src["n_layer"],
        n_head=cfg_src["n_head"],
        n_embd=cfg_src["n_embd"],
        dropout=cfg_src.get("dropout", 0.0),
        use_checkpoint=cfg_src.get("use_checkpoint", False),
        label_smoothing=cfg_src.get("label_smoothing", 0.0),
    )

    # 3) load weights; strict=True should work when arch matches
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("[state] missing:", missing)
    print("[state] unexpected:", unexpected)

    model.to(dev()).eval()

if __name__ == "__main__":
    main()
