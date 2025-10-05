"""Collect model artifacts for downstream analysis.

Usage:
    python -m scripts.collect_artifacts_yaml RUN_ID path/to/config.yaml

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
import torch.nn.functional as F
import yaml

from ._shared import (
    ArtifactError,
    ModelSpec,
    RUNS_DIR,
    build_model,
    compute_bincount,
    detect_model_type,
    ensure_run_layout,
    ensure_numpy,
    write_meta,
)

BATCH_EVAL_SIZE = 16
ATTN_DTYPE = np.float32


def _load_yaml(yaml_path: Path) -> Mapping[str, object]:
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")
    with yaml_path.open("r") as fh:
        cfg = yaml.safe_load(fh) or {}
    if not isinstance(cfg, Mapping):
        raise ValueError(f"Config at {yaml_path} is not a mapping")
    return cfg


def _resolve_path(base: Path, maybe_path: Optional[str]) -> Optional[Path]:
    if not maybe_path:
        return None
    candidate = Path(maybe_path)
    if not candidate.is_absolute():
        candidate = (base / candidate).resolve()
    return candidate


def _find_checkpoint(out_dir: Path) -> Path:
    for name in ("best.pt", "last.pt"):
        candidate = out_dir / name
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"No checkpoint found in {out_dir} (expected best.pt or last.pt)")


def _maybe_copy(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copy2(src, dst)


def _find_token_file(out_dir: Path, yaml_dir: Path, cfg: Mapping[str, object]) -> Optional[Path]:
    candidate_keys = [
        "itos_path",
        "tokenizer_path",
        "token_map",
        "token_file",
    ]
    for key in candidate_keys:
        val = cfg.get(key)
        if isinstance(val, str):
            path = _resolve_path(yaml_dir, val)
            if path and path.exists():
                return path
    for parent in {out_dir, yaml_dir}:
        if parent is None:
            continue
        cand = parent / "itos.txt"
        if cand.exists():
            return cand
    return None


def _load_val_npz(cfg: Mapping[str, object], yaml_dir: Path) -> Optional[Path]:
    keys = ["val_npz", "val_path", "validation_npz", "val_dataset"]
    for key in keys:
        if key in cfg and isinstance(cfg[key], str):
            path = _resolve_path(yaml_dir, str(cfg[key]))
            if path and path.exists():
                return path
    block_size = cfg.get("block_size")
    if block_size is None:
        return None
    data_dir = cfg.get("data_dir", "data/processed")
    data_path = _resolve_path(yaml_dir, str(data_dir))
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
    model_type = cfg.get("model_type")
    if not model_type:
        model_type = detect_model_type(state_dict)
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

    out_dir = cfg.get("out_dir")
    if not out_dir:
        raise ValueError("Config must specify out_dir")
    out_dir = _resolve_path(yaml_dir, str(out_dir))
    if out_dir is None or not out_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {out_dir}")

    checkpoint_path = _find_checkpoint(out_dir)

    weights_dst = run_dir / "weights.pt"
    if weights_dst.exists() or weights_dst.is_symlink():
        weights_dst.unlink()
    weights_dst.symlink_to(checkpoint_path)

    for optional_name in ("motif_clusters.npz", "one_cds__best.tsv"):
        src = out_dir / optional_name
        dst = run_dir / optional_name
        if src.exists():
            shutil.copy2(src, dst)

    token_path = _find_token_file(out_dir, yaml_dir, cfg)
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

    val_npz = _load_val_npz(merged_cfg, yaml_dir)
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
            "pos_embeddings": ensure_numpy(getattr(model, "pos_emb", torch.zeros(0))).astype(np.float32)
            if hasattr(model, "pos_emb")
            else np.zeros((0,), dtype=np.float32),
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

