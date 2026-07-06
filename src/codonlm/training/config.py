import os
import sys
import json
from pathlib import Path
from typing import Optional, Tuple

RUN_ID_ENV = "RUN_ID"

def write_meta(run_dir: Path, meta: dict) -> None:
    meta_path = run_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    try:
        from scripts.generate_run_summaries import generate_summary
        generate_summary(run_dir.parent)
    except Exception as e:
        print(f"[warning] Failed to generate summary.md: {e}", file=sys.stderr)


def _ensure_path_list(arg_value, cfg_value, key: str):
    source = arg_value if arg_value is not None else cfg_value
    if source is None:
        raise ValueError(f"Missing {key} specification (provide in config or CLI)")
    if isinstance(source, (str, os.PathLike)):
        return [str(source)]
    if isinstance(source, (list, tuple)):
        return [str(p) for p in source]
    raise TypeError(f"Unsupported {key} type: {type(source)}")


def _normalize_run_id(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    run_id = str(value).strip()
    return run_id or None


def _auto_run_id(cfg: dict, config_path: Optional[str]) -> str:
    from datetime import date
    today = date.today().strftime("%Y-%m-%d")
    tag = "run"
    if config_path:
        stem = Path(config_path).stem
        tag = stem.split("_", 1)[0] if "_" in stem else stem
    return f"{today}_{tag}_{int(cfg.get('n_layer',0))}L{int(cfg.get('n_head',0))}H_d{int(cfg.get('n_embd',0))}_e{int(cfg.get('epochs',0))}"


def _prepare_output_dirs(base_ckpt_dir: str, base_scores_dir: str, run_id: Optional[str]) -> Tuple[Path, Path]:
    if run_id:
        runs_dir = Path("runs") / run_id
        ckpt_root = runs_dir / "checkpoints"
        scores_root = runs_dir / "scores"
    else:
        ckpt_root = Path(base_ckpt_dir)
        scores_root = Path(base_scores_dir)

    ckpt_root.mkdir(parents=True, exist_ok=True)
    scores_root.mkdir(parents=True, exist_ok=True)
    return ckpt_root, scores_root


def _normalize_offset_weights(offsets, weights_cfg=None):
    offsets = [int(offset) for offset in offsets]
    if not offsets:
        return {}
    if weights_cfg is None:
        return {offset: 1.0 / len(offsets) for offset in offsets}
    if isinstance(weights_cfg, dict):
        return {offset: float(weights_cfg.get(offset, weights_cfg.get(str(offset), 0.0))) for offset in offsets}
    if isinstance(weights_cfg, (list, tuple)):
        if len(weights_cfg) != len(offsets):
            raise ValueError("multi_offset_weights list must match multi_offset_targets length")
        return {offset: float(weight) for offset, weight in zip(offsets, weights_cfg)}
    scalar = float(weights_cfg)
    return {offset: scalar for offset in offsets}
