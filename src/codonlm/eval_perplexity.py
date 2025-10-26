#!/usr/bin/env python3
"""
Compute validation perplexity for a saved checkpoint on the packed val set.
"""

import argparse, math, torch, numpy as np
from torch.utils.data import DataLoader, Dataset
from .model_tiny_gpt import TinyGPT, Cfg

def _detect_model_type(sd: dict) -> str:
    keys = list(sd.keys())
    if any(".attn.qkv.weight" in k or ".attn.attn_mask" in k for k in keys):
        return "tiny_gpt"
    if any(".attn.key.weight" in k or ".attn.mask" in k for k in keys):
        return "tiny_gpt_v2"
    return "tiny_gpt"

# --- utility: device + legacy ckpt conversion + layer inference ---
def dev():
    return "mps" if torch.backends.mps.is_available() else "cpu"

import torch

def _is_legacy_state(sd: dict) -> bool:
    return any(".attn.query.weight" in k for k in sd.keys()) or any(k.endswith(".attn.mask") for k in sd.keys())

def _convert_legacy_state(sd_old: dict) -> dict:
    sd_new = {}
    for k, v in sd_old.items():
        # drop old attention mask buffer; new model creates attn_mask itself
        if k.endswith(".attn.mask"):
            continue

        # fuse query/key/value -> qkv (WEIGHTS)
        if k.endswith(".attn.query.weight"):
            base = k[:-len(".attn.query.weight")]
            qw = sd_old[k]
            kw = sd_old[base + ".attn.key.weight"]
            vw = sd_old[base + ".attn.value.weight"]
            sd_new[base + ".attn.qkv.weight"] = torch.cat([qw, kw, vw], dim=0)
            continue

        # drop separate q/k/v weights (handled above)
        if (".attn.key.weight" in k) or (".attn.value.weight" in k):
            continue

        # drop old biases if new layers use bias=False on attn.proj or qkv
        if (".attn.query.bias" in k) or (".attn.key.bias" in k) or (".attn.value.bias" in k) or k.endswith(".attn.proj.bias"):
            continue

        # keep everything else
        sd_new[k] = v
    return sd_new

def _infer_n_layers_from_state(sd: dict) -> int:
    max_idx = -1
    for k in sd.keys():
        if k.startswith("blocks."):
            try:
                i = int(k.split('.')[1])
                if i > max_idx: max_idx = i
            except: pass
    return max_idx + 1 if max_idx >= 0 else 0

class PackedDataset(Dataset):
    def __init__(self, npz_path):
        d = np.load(npz_path)
        self.X = torch.from_numpy(d["X"]).long()
        self.Y = torch.from_numpy(d["Y"]).long()
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.Y[i]

def dev():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val_npz", required=True)
    args = ap.parse_args()

    # 1) load checkpoint
    state = torch.load(args.ckpt, map_location="cpu")
    sd = state.get("model", state)
    cfg_saved = state.get("cfg", None)

    model_type = _detect_model_type(sd)
    if model_type == "tiny_gpt_v2":
        from .model_tiny_gpt_v2 import TinyGPTv2
        cfg_src = cfg_saved or {
            "vocab_size": getattr(args, "vocab_size", 69),
            "block_size": getattr(args, "block_size", 256),
            "n_layer": getattr(args, "n_layer", 2),
            "n_head": getattr(args, "n_head", 4),
            "n_embd": getattr(args, "n_embd", 128),
            "dropout": getattr(args, "dropout", 0.0),
        }
        model = TinyGPTv2(
            vocab_size=cfg_src["vocab_size"],
            block_size=cfg_src["block_size"],
            n_layer=cfg_src["n_layer"],
            n_head=cfg_src["n_head"],
            n_embd=cfg_src["n_embd"],
            dropout=cfg_src.get("dropout", 0.0),
        )
    else:
        # v1
        if cfg_saved is not None:
            mconf = Cfg(
                vocab_size = cfg_saved["vocab_size"],
                n_layer    = cfg_saved["n_layer"],
                n_head     = cfg_saved["n_head"],
                n_embd     = cfg_saved["n_embd"],
                block_size = cfg_saved["block_size"],
                dropout    = cfg_saved.get("dropout", 0.0),
            )
        else:
            mconf = Cfg(
                vocab_size = args.vocab_size,
                n_layer    = args.n_layer,
                n_head     = args.n_head,
                n_embd     = args.n_embd,
                block_size = args.block_size,
                dropout    = getattr(args, "dropout", 0.0),
            )
        model = TinyGPT(mconf)

    # 3) load weights; strict=True should work when arch matches
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("[state] missing:", missing)
    print("[state] unexpected:", unexpected)

    model.to(dev()).eval()

if __name__ == "__main__":
    main()
