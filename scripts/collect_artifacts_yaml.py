"""Collect model artifacts for downstream analysis.

Usage:
    python -m scripts.collect_artifacts_yaml RUN_ID configs/config.yaml

The script loads the YAML configuration, locates the trained checkpoint,
symlinks the weights into ``runs/<run_id>/weights.pt`` and saves a compact
``artifacts.npz`` bundle with token/positional embeddings, validation
statistics, logits, probabilities, and attention tensors. Optional files
(``motif_clusters.npz`` and ``one_cds__best.tsv``) are copied when present.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import os

from ._shared import (
    ArtifactError,
    ModelSpec,
    RUNS_DIR,
    build_model,
    compute_bincount,
    ensure_run_layout,
    ensure_numpy,
    write_meta,
)

BATCH_EVAL_SIZE = 16
ATTN_DTYPE = np.float32

REPO_MARKERS = {".git", "pyproject.toml", "setup.cfg", "setup.py"}

def _find_repo_root(start: Path) -> Path:
    cur = start
    while True:
        if any((cur / m).exists() for m in REPO_MARKERS):
            return cur
        if cur.parent == cur:
            # Fallback: if we never found a marker, assume configs/ is directly under root
            return start.parent
        cur = cur.parent

def _resolve_from(base: Path, maybe_path: Optional[str]) -> Optional[Path]:
    """Expand ~ and $VARS, then make relative to base."""
    if not maybe_path:
        return None
    p = Path(os.path.expanduser(os.path.expandvars(maybe_path)))
    if p.is_absolute():
        return p
    return (base / p).resolve()


def _extract_positional_embeddings(model: torch.nn.Module) -> np.ndarray:
    """
    Return positional embeddings as float32 numpy array if present, else empty array.
    Supports:
      - Tensor buffer (e.g., model.pos_emb is a Tensor)
      - nn.Embedding module (use .weight)
      - Modules with a .weight tensor
    Also tries a few common attribute names as fallbacks.
    """
    cand_names = ["pos_emb", "positional_embedding", "position_embeddings", "wpe"]
    for name in cand_names:
        if not hasattr(model, name):
            continue
        pe = getattr(model, name)
        # Case 1: direct Tensor
        if torch.is_tensor(pe):
            return ensure_numpy(pe).astype(np.float32)
        # Case 2: nn.Embedding or anything with a .weight Tensor
        if isinstance(pe, nn.Module) and hasattr(pe, "weight") and torch.is_tensor(pe.weight):
            return ensure_numpy(pe.weight).astype(np.float32)
    # Nothing found
    return np.zeros((0,), dtype=np.float32)


def _load_yaml(yaml_path: Path) -> Mapping[str, object]:
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")
    with yaml_path.open("r") as fh:
        cfg = yaml.safe_load(fh) or {}
    if not isinstance(cfg, Mapping):
        raise ValueError(f"Config at {yaml_path} is not a mapping")
    return cfg

def _find_checkpoint(check_dir: Path, run_id: Optional[str] = None) -> Path:
    """
    Look for best.pt/last.pt in either:
    - check_dir (flat layout), or
    - check_dir/<run_id>/ (per-run layout)
    """
    search_roots = []
    if run_id:
        run_sub = check_dir / run_id
        if run_sub.exists():
            search_roots.append(run_sub)
    search_roots.append(check_dir)

    # fallback: allow common alternate directory names (legacy layouts)
    alt_dirs = []
    # Variants with provided run_id
    if run_id:
        alt_dirs.extend([
            check_dir.parent / f"{check_dir.name}_{run_id}",              # e.g., outputs/checkpoints_<RUN_ID>
            check_dir.parent / f"{check_dir.name}_tiny" / run_id,        # e.g., outputs/checkpoints_tiny/<RUN_ID>
            Path("outputs") / f"{check_dir.name}" / run_id,             # e.g., outputs/checkpoints/<RUN_ID>
            Path("outputs") / f"{check_dir.name}_tiny" / run_id,        # e.g., outputs/checkpoints_tiny/<RUN_ID>
        ])
    # Flat directories without run_id (v1 layout like outputs/checkpoints_tiny/best.pt)
    alt_dirs.extend([
        check_dir.parent / f"{check_dir.name}_tiny",                      # e.g., outputs/checkpoints_tiny
        Path("outputs") / f"{check_dir.name}_tiny",                     # e.g., outputs/checkpoints_tiny
    ])
    for alt in alt_dirs:
        if alt.exists() and alt not in search_roots:
            search_roots.append(alt)

    for root in search_roots:
        for name in ("best.pt", "last.pt"):
            cand = root / name
            if cand.exists():
                return cand.resolve()

    places = ", ".join(str(r) for r in search_roots)
    raise FileNotFoundError(f"No checkpoint found (looked for best.pt/last.pt) under: {places}")


def _maybe_copy(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copy2(src, dst)


def _find_token_file(out_dir: Path, repo_root: Path, cfg: Mapping[str, object], run_dir: Optional[Path] = None) -> Optional[Path]:
    candidate_keys = ["itos_path", "tokenizer_path", "token_map", "token_file"]
    for key in candidate_keys:
        val = cfg.get(key)
        if isinstance(val, str):
            path = _resolve_from(repo_root, val)
            if path and path.exists():
                return path
    # common fallbacks
    for parent in {out_dir, repo_root}:
        if parent is None:
            continue
        cand = parent / "itos.txt"
        if cand.exists():
            return cand
    if run_dir:
        manifest_path = run_dir / "combined_manifest.json"
        if manifest_path.exists():
            try:
                combined = json.loads(manifest_path.read_text())
                for ds in combined.get("datasets", []):
                    itos_path = ds.get("itos")
                    if isinstance(itos_path, str):
                        itos = Path(itos_path)
                        if not itos.is_absolute():
                            itos = (repo_root / itos).resolve()
                        if itos.exists():
                            return itos
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[warn] failed to parse {manifest_path}: {exc}")
    return None


def _load_val_npz(cfg: Mapping[str, object], repo_root: Path) -> Optional[Path]:
    keys = ["val_npz", "val_path", "validation_npz", "val_dataset"]
    for key in keys:
        if key not in cfg:
            continue
        value = cfg[key]
        paths: Iterable[str]
        if isinstance(value, (list, tuple)):
            paths = [str(p) for p in value]
        else:
            paths = [str(value)]
        for item in paths:
            path = _resolve_from(repo_root, item)
            if path and path.exists():
                return path
    block_size = cfg.get("block_size")
    if block_size is None:
        return None
    data_dir = cfg.get("data_dir", "data/processed")
    data_path = _resolve_from(repo_root, str(data_dir))
    if data_path is None:
        return None
    candidate = data_path / f"val_bs{block_size}.npz"
    if candidate.exists():
        return candidate
    return None


def _load_npz_dataset(npz_path: Path) -> Dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as data:
        arrays = {k: np.asarray(data[k]) for k in data.files}
    return arrays


def _infer_spec_from_state(cfg: Mapping[str, object], state_dict: Mapping[str, torch.Tensor]) -> ModelSpec:
    model_type = cfg.get("model_type", "tiny_gpt")
    required_keys = ["vocab_size", "block_size", "n_layer", "n_head", "n_embd"]
    missing = [k for k in required_keys if k not in cfg]
    if missing:
        raise ArtifactError(f"Missing model parameters in config: {missing}")
    spec = ModelSpec(
        model_type=model_type,
        vocab_size=int(cfg["vocab_size"]),
        block_size=int(cfg["block_size"]),
        n_layer=int(cfg["n_layer"]),
        n_head=int(cfg["n_head"]),
        n_embd=int(cfg["n_embd"]),
        dropout=float(cfg.get("dropout", 0.0)),
        use_checkpoint=bool(cfg.get("use_checkpoint", False)),
        label_smoothing=float(cfg.get("label_smoothing", 0.0)),
    )
    return spec


def _capture_attention(model: torch.nn.Module, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    attn_maps: Dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer_idx: int):
        def _hook(module: torch.nn.Module, module_inputs: Tuple[torch.Tensor, ...], module_output: torch.Tensor):
            x = module_inputs[0]
            with torch.no_grad():
                if hasattr(module, "qkv"):
                    qkv = module.qkv(x)
                    q, k, v = qkv.chunk(3, dim=-1)
                    B, T, C = q.shape
                    H = module.n_head
                    q = q.view(B, T, H, C // H).transpose(1, 2)
                    k = k.view(B, T, H, C // H).transpose(1, 2)
                    att = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
                    mask = module.attn_mask[:, :, :T, :T]
                    att = att.masked_fill(mask == 0, float("-inf"))
                    att = torch.softmax(att, dim=-1)
                elif hasattr(module, "key"):
                    B, T, _ = x.shape
                    n_head = module.n_head
                    k = module.key(x).view(B, T, n_head, -1).transpose(1, 2)
                    q = module.query(x).view(B, T, n_head, -1).transpose(1, 2)
                    att = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
                    mask = module.mask[:, :, :T, :T]
                    att = att.masked_fill(mask == 0, float("-inf"))
                    att = torch.softmax(att, dim=-1)
                else:
                    raise RuntimeError("Unknown attention module; cannot capture attention map")
            attn_maps[layer_idx] = att.detach().cpu()

        return _hook

    for idx, block in enumerate(getattr(model, "blocks", [])):
        attn = getattr(block, "attn", None)
        if attn is None:
            continue
        handles.append(attn.register_forward_hook(make_hook(idx)))

    with torch.no_grad():
        outputs = model(inputs)
        logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs["logits"]

    for handle in handles:
        handle.remove()

    if attn_maps:
        ordered = [attn_maps[i] for i in sorted(attn_maps.keys())]
        attn_tensor = torch.stack(ordered, dim=0)
    else:
        attn_tensor = torch.zeros(0)
    return logits, attn_tensor


def _evaluate_perplexity(model: torch.nn.Module, inputs: np.ndarray, targets: np.ndarray, vocab: int) -> Optional[float]:
    if inputs.size == 0 or targets.size == 0:
        return None
    device = next(model.parameters()).device
    batch = BATCH_EVAL_SIZE
    total_loss = 0.0
    total_tokens = 0
    for start in range(0, inputs.shape[0], batch):
        end = min(start + batch, inputs.shape[0])
        xb = torch.from_numpy(inputs[start:end]).to(device=device, dtype=torch.long)
        yb = torch.from_numpy(targets[start:end]).to(device=device, dtype=torch.long)
        with torch.no_grad():
            outputs = model(xb, yb)
            if isinstance(outputs, tuple):
                loss = outputs[1]
            elif isinstance(outputs, Mapping):
                loss = outputs.get("loss")
                if loss is None:
                    logits = outputs["logits"]
                    loss = F.cross_entropy(logits.view(-1, vocab), yb.view(-1))
            else:
                raise RuntimeError("Unexpected model output during evaluation")
        total_loss += float(loss.item()) * (end - start)
        total_tokens += (end - start)
    if total_tokens == 0:
        return None
    mean_loss = total_loss / total_tokens
    return float(math.exp(min(50.0, mean_loss)))


def _save_artifacts(run_dir: Path, arrays: Dict[str, np.ndarray]) -> None:
    out_path = run_dir / "artifacts.npz"
    np.savez_compressed(out_path, **arrays)


def main(argv: Optional[Iterable[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("config")
    args = ap.parse_args(argv)

    run_paths = ensure_run_layout(args.run_id)
    run_dir = run_paths["run"]
    yaml_path = Path(args.config).resolve()
    cfg = _load_yaml(yaml_path)
    yaml_dir = yaml_path.parent
    repo_root = _find_repo_root(yaml_dir)

    out_dir_cfg = cfg.get("out_dir") or cfg.get("checkpoints_dir") or "outputs/checkpoints"
    out_dir = _resolve_from(repo_root, str(out_dir_cfg))
    if out_dir is None or not out_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {out_dir}")

    checkpoint_path = _find_checkpoint(out_dir, run_id=args.run_id)

    weights_dst = run_dir / "weights.pt"
    if weights_dst.exists() or weights_dst.is_symlink():
        weights_dst.unlink()
    weights_dst.symlink_to(checkpoint_path)

    for optional_name in ("motif_clusters.npz", "one_cds__best.tsv"):
        src = out_dir / optional_name
        dst = run_dir / optional_name
        if src.exists():
            shutil.copy2(src, dst)

    token_path = _find_token_file(out_dir, repo_root, cfg, run_dir)
    vocab_size = int(cfg.get("vocab_size", 0) or 0)
    if token_path and token_path.exists():
        tokens = [line.strip() for line in token_path.read_text().splitlines() if line.strip()]
    else:
        tokens = [f"tok_{i}" for i in range(vocab_size)]
    (run_dir / "itos.txt").write_text("\n".join(tokens) + "\n")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, Mapping) and "model" in checkpoint:
        state_dict = checkpoint["model"]
        ckpt_cfg = checkpoint.get("cfg", {}) if isinstance(checkpoint.get("cfg"), Mapping) else {}
    else:
        state_dict = checkpoint
        ckpt_cfg = {}
    merged_cfg = dict(cfg)
    merged_cfg.update(ckpt_cfg)

    spec = _infer_spec_from_state(merged_cfg, state_dict)
    model = build_model(spec)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    val_npz = _load_val_npz(merged_cfg, repo_root)
    arrays: Dict[str, np.ndarray] = {}
    token_counts = None
    first_token_counts = None
    val_inputs = None
    val_targets = None

    if val_npz and val_npz.exists():
        data = _load_npz_dataset(val_npz)
        if "X" in data:
            val_inputs = np.asarray(data["X"], dtype=np.int64)
            token_counts = compute_bincount(val_inputs, spec.vocab_size)
            if val_inputs.size > 0:
                first_token_counts = compute_bincount(val_inputs[:, :1], spec.vocab_size)
        if "Y" in data:
            val_targets = np.asarray(data["Y"], dtype=np.int64)
        if val_targets is None and val_inputs is not None:
            val_targets = np.roll(val_inputs, -1, axis=1)
        arrays["val_inputs"] = val_inputs[:32].copy() if val_inputs is not None else np.zeros((0,), dtype=np.int64)
        arrays["val_targets"] = val_targets[:32].copy() if val_targets is not None else np.zeros((0,), dtype=np.int64)
    else:
        arrays["val_inputs"] = np.zeros((0,), dtype=np.int64)
        arrays["val_targets"] = np.zeros((0,), dtype=np.int64)

    batch_inputs = arrays["val_inputs"]
    logits_arr = np.zeros((0,), dtype=np.float32)
    probs_arr = np.zeros((0,), dtype=np.float32)
    attn_arr = np.zeros((0,), dtype=ATTN_DTYPE)

    if batch_inputs.size > 0:
        batch_tensor = torch.from_numpy(batch_inputs[:4]).long()
        logits, attn = _capture_attention(model, batch_tensor)
        probs = torch.softmax(logits, dim=-1)
        logits_arr = logits.detach().cpu().numpy()
        probs_arr = probs.detach().cpu().numpy()
        attn_arr = attn.detach().cpu().numpy().astype(ATTN_DTYPE)

    arrays.update(
        {
            "token_embeddings": ensure_numpy(model.tok_emb.weight).astype(np.float32),
            "pos_embeddings": _extract_positional_embeddings(model),
            "logits": logits_arr,
            "probs": probs_arr,
            "attn": attn_arr,
            "token_counts": token_counts if token_counts is not None else np.zeros((vocab_size,), dtype=np.int64),
            "first_token_counts": first_token_counts if first_token_counts is not None else np.zeros((vocab_size,), dtype=np.int64),
        }
    )

    _save_artifacts(run_dir, arrays)

    val_ppl = None
    if val_inputs is not None and val_targets is not None and val_inputs.size > 0:
        val_ppl = _evaluate_perplexity(model, val_inputs, val_targets, spec.vocab_size)

    meta = {
        "run_id": args.run_id,
        "config_path": str(yaml_path),
        "out_dir": str(out_dir),
        "checkpoint_path": str(checkpoint_path),
        "val_npz": str(val_npz) if val_npz else None,
        "val_ppl": val_ppl,
        "model_spec": spec.to_dict(),
        "token_count": len(tokens),
    }
    write_meta(run_dir, meta)

    print(f"[collect] saved artifacts for {args.run_id} → {run_dir}")


if __name__ == "__main__":
    main()
