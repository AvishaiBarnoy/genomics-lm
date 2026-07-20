from __future__ import annotations

from pathlib import Path
from typing import Mapping
import hashlib

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
        use_shape_guidance=bool(cfg.get("use_shape_guidance", False)),
    )


def load_codon_model(
    run_dir: str | Path,
    device: torch.device | str = "cpu",
    ckpt_name: str = "best.pt",
) -> tuple[TinyGPT, dict, Path]:
    state_dict, cfg, ckpt_path = load_codon_checkpoint(run_dir, ckpt_name=ckpt_name)
    configured_size = cfg.get("vocab_size")
    itos_path = Path(run_dir) / "itos.txt"
    artifact_tokens = (
        [line.strip() for line in itos_path.read_text().splitlines()]
        if itos_path.exists()
        else []
    )
    artifact_size = len(artifact_tokens) if artifact_tokens else None
    embedding_rows = None
    if "tok_emb.weight" in state_dict:
        embedding_rows = int(state_dict["tok_emb.weight"].shape[0])
        cfg["vocab_size"] = embedding_rows
    elif artifact_size is not None:
        cfg["vocab_size"] = artifact_size
    output_rows = (
        int(state_dict["head.weight"].shape[0])
        if "head.weight" in state_dict
        else None
    )
    if embedding_rows is not None and output_rows is not None and embedding_rows != output_rows:
        raise RuntimeError(
            f"Legacy checkpoint has incompatible embedding rows={embedding_rows} "
            f"and output rows={output_rows}: {ckpt_path}"
        )
    cfg["vocabulary_compatibility"] = {
        "mode": "legacy_checkpoint_inference",
        "configured_size": configured_size,
        "embedding_rows": embedding_rows,
        "output_rows": output_rows,
        "artifact_path": str(itos_path) if itos_path.exists() else None,
        "artifact_size": artifact_size,
        "artifact_sha256": (
            hashlib.sha256(itos_path.read_bytes()).hexdigest()
            if itos_path.exists()
            else None
        ),
        "legacy_adaptation": any(
            value is not None and value != cfg["vocab_size"]
            for value in (configured_size, artifact_size)
        ),
    }
    model = build_codon_model_from_cfg(cfg)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model, cfg, ckpt_path
