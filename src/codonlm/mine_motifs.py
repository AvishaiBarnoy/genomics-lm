#!/usr/bin/env python3
"""
Extract token embeddings for sliding windows and cluster them to discover recurring "motifs".

Method:
- Feed sequences through the model, grab hidden states (pre-head features).
- Average-pool embeddings over k=9 tokens (≈ 9-codon motif).
- KMeans cluster → inspect cluster consensus.
"""

import argparse
import torch
import numpy as np
from sklearn.cluster import KMeans
from .model_tiny_gpt import TinyGPT, Cfg
import inspect  # <-- import here, NOT inside main()

def dev():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

def to_idx_tensor(x_like, device, block_size):
    """Build a safe (B,T) Long tensor for Embedding on MPS/CPU."""
    if isinstance(x_like, np.ndarray):
        x_cpu = torch.from_numpy(np.ascontiguousarray(x_like, dtype=np.int64))
    elif isinstance(x_like, (list, tuple)):
        x_cpu = torch.tensor(x_like, dtype=torch.long)
    elif isinstance(x_like, torch.Tensor):
        x_cpu = x_like.to("cpu", copy=True).contiguous().long()
    else:
        raise TypeError(f"Unsupported type for indices: {type(x_like)}")
    if x_cpu.ndim == 1:
        x_cpu = x_cpu.unsqueeze(0)  # (1,T)
    if x_cpu.size(1) > block_size:
        x_cpu = x_cpu[:, -block_size:]
    return x_cpu.to(device, non_blocking=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--npz", required=True, help="train npz to sample windows from")
    ap.add_argument("--k", type=int, default=9)
    ap.add_argument("--clusters", type=int, default=100)
    ap.add_argument("--samples", type=int, default=20000)
    args = ap.parse_args()

    # ---- load checkpoint and rebuild model config (match layers/dims) ----
    state = torch.load(args.ckpt, map_location="cpu")
    sd = state.get("model", state)
    cfg_saved = state.get("cfg", None)

    def detect(sd: dict) -> str:
        ks = list(sd.keys())
        if any(".attn.qkv.weight" in k or ".attn.attn_mask" in k for k in ks):
            return "tiny_gpt"
        if any(".attn.key.weight" in k or ".attn.mask" in k for k in ks):
            return "tiny_gpt_v2"
        return "tiny_gpt"

    model_type = detect(sd)
    if model_type == "tiny_gpt_v2":
        from .model_tiny_gpt_v2 import TinyGPTv2
        cfg_src = cfg_saved or {"vocab_size":69,"block_size":256,"n_layer":2,"n_head":4,"n_embd":128,"dropout":0.0}
        model = TinyGPTv2(cfg_src["vocab_size"], cfg_src["block_size"], n_layer=cfg_src["n_layer"], n_head=cfg_src["n_head"], n_embd=cfg_src["n_embd"], dropout=cfg_src.get("dropout",0.0))
    else:
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
            mconf = Cfg(vocab_size=69, n_layer=2, n_head=4, n_embd=128, block_size=256, dropout=0.1)
        model = TinyGPT(mconf)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("[state] missing:", missing)
    print("[state] unexpected:", unexpected)

    device = dev()
    model.to(device).eval()

    print("Using TinyGPT from:", inspect.getsourcefile(TinyGPT))
    # derive a readable config summary regardless of v1/v2
    model_cfg = {}
    if cfg_saved:
        model_cfg = dict(cfg_saved)
    else:
        # fallback to reading attributes
        model_cfg = {
            "vocab_size": getattr(getattr(model, "tok_emb", None), "num_embeddings", None),
            "block_size": getattr(model, "block_size", getattr(getattr(model, "pos_emb", None), "num_embeddings", None)),
            "n_embd": getattr(getattr(model, "ln_f", None), "normalized_shape", [None])[0],
        }
    print("Model cfg:", model_cfg)
    print("Model device:", next(model.parameters()).device)
    print("MPS available:", torch.backends.mps.is_available())

    # ---- data ----
    d = np.load(args.npz, allow_pickle=False)
    arr = np.asarray(d["X"])  # (N, T)

    # ---- capture pre-head features via forward hook (output of ln_f) ----
    feats_buf = []
    def hook_ln_f(module, inp, out):
        # out shape: (B, T, n_embd)
        feats_buf.append(out.detach().cpu())

    h = model.ln_f.register_forward_hook(hook_ln_f)

    # ---- collect pooled embeddings over k-length windows ----
    k = int(args.k)
    vecs = []
    with torch.no_grad():
        n_samples = min(args.samples, arr.shape[0])
        for i in range(n_samples):
            # safe indices tensor on correct device
            block_size = (
                getattr(model, "block_size", None)
                or getattr(getattr(model, "pos_emb", None), "num_embeddings", None)
                or model_cfg.get("block_size", 256)
            )
            x = to_idx_tensor(arr[i], device, int(block_size))  # (1, T')
            feats_buf.clear()
            logits, _ = model(x)          # runs forward, hook fills feats_buf
            # take the last captured features
            h_last = feats_buf[-1].squeeze(0)   # (T', n_embd)
            Tprime = h_last.size(0)
            if Tprime < k: 
                continue
            # average-pool over sliding windows of length k
            # stack mean over time dimension (dim=0), result: (T'-k+1, n_embd)
            win_means = torch.stack([h_last[t:t+k].mean(dim=0) for t in range(0, Tprime - k + 1)], dim=0)
            vecs.append(win_means.numpy())

    h.remove()  # remove hook

    if not vecs:
        raise RuntimeError("No windows collected; check sequence lengths and k.")

    vecs = np.concatenate(vecs, axis=0)  # (num_windows, n_embd)
    print("[motifs] collected windows:", vecs.shape[0])

    # ---- KMeans clustering ----
    km = KMeans(n_clusters=args.clusters, n_init="auto").fit(vecs)
    print("[motifs] inertia:", km.inertia_)
    np.savez_compressed("outputs/motif_clusters.npz", centers=km.cluster_centers_, labels=km.labels_)
    print("[motifs] saved outputs/motif_clusters.npz")

if __name__ == "__main__":
    main()
