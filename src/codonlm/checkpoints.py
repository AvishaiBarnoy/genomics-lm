from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch

from .model_tiny_gpt import TinyGPT


def load_codon_checkpoint(run_dir: str | Path, ckpt_name: str = "best.pt") -> tuple[dict, dict, Path]:
    run_path = Path(run_dir)
    candidates = [
        run_path / "checkpoints" / ckpt_name,
        run_path / ckpt_name,
    ]
    if ckpt_name == "best.pt":
        candidates.extend([
            run_path / "checkpoints" / "weights.pt",
            run_path / "weights.pt",
        ])
    for path in candidates:
        if path.exists():
            state = torch.load(path, map_location="cpu")
            if isinstance(state, Mapping) and "model" in state:
                return state["model"], dict(state.get("cfg", {})), path
            return state, {}, path
    raise FileNotFoundError(f"{ckpt_name} not found under {run_path}")


def build_codon_model_from_cfg(cfg: dict) -> TinyGPT:
    required = ["vocab_size", "block_size", "n_layer", "n_head", "n_embd"]
    missing = [key for key in required if key not in cfg]
    if missing:
        raise RuntimeError(f"Checkpoint config missing fields: {missing}")
    return TinyGPT(
        vocab_size=int(cfg["vocab_size"]),
        block_size=int(cfg["block_size"]),
        n_layer=int(cfg["n_layer"]),
        n_head=int(cfg["n_head"]),
        n_embd=int(cfg["n_embd"]),
        dropout=float(cfg.get("dropout", 0.0)),
        use_checkpoint=False,
        label_smoothing=float(cfg.get("label_smoothing", 0.0)),
        sep_id=(3 if bool(cfg.get("sep_mask_enabled", True)) else None),
        tie_embeddings=bool(cfg.get("tie_embeddings", True)),
        n_kv_head=int(cfg.get("n_kv_head")) if cfg.get("n_kv_head") is not None else None,
        use_sdpa=bool(cfg.get("use_sdpa", False)),
        termination_aux=bool(cfg.get("termination_loss_enabled", cfg.get("termination_aux", False))),
        termination_n_classes=int(cfg.get("termination_n_classes", 5)),
        multi_offset_targets=cfg.get("multi_offset_targets", None),
        use_swiglu=bool(cfg.get("use_swiglu", False)),
        use_rope=bool(cfg.get("use_rope", False)),
    )


def load_codon_model(
    run_dir: str | Path,
    device: torch.device | str = "cpu",
    ckpt_name: str = "best.pt",
) -> tuple[TinyGPT, dict, Path]:
    state_dict, cfg, ckpt_path = load_codon_checkpoint(run_dir, ckpt_name=ckpt_name)
    model = build_codon_model_from_cfg(cfg)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model, cfg, ckpt_path
