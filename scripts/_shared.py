"""Shared utilities for analysis scripts.

These helpers centralize file-system layout, artifact loading, model
reconstruction, and token-mapping utilities so that downstream analysis
scripts stay small and consistent. All functions are intentionally
side-effect free (other than simple path creation) so they can be reused
across command-line tools.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

# Default location for processed run artifacts.
RUNS_DIR = Path("runs")


class ArtifactError(RuntimeError):
    """Raised when required artifacts are missing."""


@dataclass
class ModelSpec:
    """Lightweight description of a TinyGPT variant."""

    model_type: str
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float = 0.0
    use_checkpoint: bool = False

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ModelSpec":
        required = [
            "model_type",
            "vocab_size",
            "block_size",
            "n_layer",
            "n_head",
            "n_embd",
        ]
        missing = [k for k in required if k not in data]
        if missing:
            raise ArtifactError(f"model specification missing keys: {missing}")
        return cls(
            model_type=str(data["model_type"]),
            vocab_size=int(data["vocab_size"]),
            block_size=int(data["block_size"]),
            n_layer=int(data["n_layer"]),
            n_head=int(data["n_head"]),
            n_embd=int(data["n_embd"]),
            dropout=float(data.get("dropout", 0.0)),
            use_checkpoint=bool(data.get("use_checkpoint", False)),
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "model_type": self.model_type,
            "vocab_size": self.vocab_size,
            "block_size": self.block_size,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "n_embd": self.n_embd,
            "dropout": self.dropout,
            "use_checkpoint": self.use_checkpoint,
        }


# ----- filesystem helpers -------------------------------------------------

def ensure_run_layout(run_id: str) -> Dict[str, Path]:
    """Ensure that the standard directory layout for a run exists."""

    run_dir = RUNS_DIR / run_id
    charts_dir = run_dir / "charts"
    tables_dir = run_dir / "tables"
    for path in (run_dir, charts_dir, tables_dir):
        path.mkdir(parents=True, exist_ok=True)
    return {"run": run_dir, "charts": charts_dir, "tables": tables_dir}


def read_meta(run_dir: Path) -> Dict[str, object]:
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        raise ArtifactError(f"meta.json missing for run at {run_dir}")
    return json.loads(meta_path.read_text())


def write_meta(run_dir: Path, meta: Mapping[str, object]) -> None:
    meta_path = run_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))


def load_artifacts(run_id: str) -> Mapping[str, np.ndarray]:
    path = RUNS_DIR / run_id / "artifacts.npz"
    if not path.exists():
        raise ArtifactError(f"artifacts.npz missing for run {run_id}")
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def load_token_list(run_dir: Path) -> List[str]:
    tok_path = run_dir / "itos.txt"
    if not tok_path.exists():
        raise ArtifactError(f"itos.txt missing under {run_dir}")
    tokens = [line.strip() for line in tok_path.read_text().splitlines() if line.strip()]
    if not tokens:
        raise ArtifactError(f"itos.txt at {tok_path} is empty")
    return tokens


def stoi(tokens: Sequence[str]) -> Dict[str, int]:
    return {tok: i for i, tok in enumerate(tokens)}


# ----- model reconstruction ------------------------------------------------

def detect_model_type(state_dict: Mapping[str, torch.Tensor]) -> str:
    keys = list(state_dict.keys())
    if any("qkv" in k for k in keys):
        return "tiny_gpt"
    if any("key.weight" in k for k in keys):
        return "tiny_gpt_v2"
    raise ArtifactError("Unable to infer model type from checkpoint keys")


def build_model(spec: ModelSpec) -> torch.nn.Module:
    if spec.model_type == "tiny_gpt":
        from src.codonlm.model_tiny_gpt import Cfg, TinyGPT

        cfg = Cfg(
            vocab_size=spec.vocab_size,
            n_layer=spec.n_layer,
            n_head=spec.n_head,
            n_embd=spec.n_embd,
            block_size=spec.block_size,
            dropout=spec.dropout,
        )
        model = TinyGPT(cfg)
    elif spec.model_type == "tiny_gpt_v2":
        from src.codonlm.model_tiny_gpt_v2 import TinyGPTv2

        model = TinyGPTv2(
            vocab_size=spec.vocab_size,
            block_size=spec.block_size,
            n_layer=spec.n_layer,
            n_head=spec.n_head,
            n_embd=spec.n_embd,
            dropout=spec.dropout,
            use_checkpoint=spec.use_checkpoint,
        )
    else:
        raise ArtifactError(f"Unsupported model_type {spec.model_type}")
    model.eval()
    return model


def load_model(run_dir: Path, device: torch.device | str = "cpu") -> Tuple[torch.nn.Module, ModelSpec]:
    meta = read_meta(run_dir)
    spec = ModelSpec.from_dict(meta["model_spec"])
    weights_path = run_dir / "weights.pt"
    if not weights_path.exists():
        raise ArtifactError(f"weights.pt missing under {run_dir}")
    ckpt = torch.load(weights_path, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, Mapping) and "model" in ckpt else ckpt
    model = build_model(spec)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model, spec


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.nn.functional.softmax(x, dim=dim)


def compute_bincount(values: np.ndarray, size: int) -> np.ndarray:
    flat = values.reshape(-1)
    counts = np.bincount(flat, minlength=size)
    if counts.shape[0] > size:
        counts = counts[:size]
    return counts.astype(np.int64)


def ensure_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    return x.detach().cpu().numpy()


