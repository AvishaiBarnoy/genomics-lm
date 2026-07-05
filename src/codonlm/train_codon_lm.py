#!/usr/bin/env python3
"""
Training with:
- MPS autocast float16 (Apple Silicon)
- optional gradient checkpointing
- cosine or plateau LR schedules with early stopping
- CSV logging

Config highlights:
  amp: true/false
  use_checkpoint: true/false
  optimizer: "adamw" or "adafactor"
  scheduler: "cosine" (default) or "plateau"
  warmup_steps: linear warmup period
  min_lr: final LR for cosine schedule / floor for plateau
  early_stop_patience: epochs without improvement before stopping
  log_csv: where to append training curves (relative to scores_dir by default)
"""

import argparse
import yaml
import math
import csv
import time
import torch
import json
import os
import sys
import logging
import shutil
from pathlib import Path
from typing import Optional, Tuple
from torch.utils.data import DataLoader
from .model_tiny_gpt import TinyGPT
from .data_loading import (
    PackedDataset,  # noqa: F401 - re-exported for legacy tests/imports
    build_codon_lm_dataloaders,
    build_codon_lm_datasets,
    dataset_length_audit,
)
from .replay import GeneratedTerminationReplayDataset
from src.training.runtime import (
    PeriodicCheckpointPolicy,
    RunLogger,
    WallTimeLimitException,
    WallTimer,
    default_device,
    save_checkpoint_atomic,
)

RUN_ID_ENV = "RUN_ID"
PAD_ID = 0
DEFAULT_BOUNDARY_IDS = (2, 3)  # <EOS_CDS>, <SEP>


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
    try:
        from . import __package__  # noqa: F401
    except Exception:
        pass
    # Build RUN_ID like YYYY-MM-DD_<model>_<nL><nH>_d<n_embd>_e<epochs>
    from datetime import date
    from pathlib import Path
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


def _read_itos(path_value: Optional[str], base_dir: Path | None = None) -> list[str] | None:
    if not path_value:
        return None
    path = Path(str(path_value))
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    if not path.exists():
        return None
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _load_transfer_state_dict(
    model: torch.nn.Module,
    source_state: dict,
    *,
    source_itos: list[str] | None = None,
    target_itos: list[str] | None = None,
) -> dict:
    """Load a checkpoint into a model, allowing tokenizer/vocab expansion.

    Exact-shape tensors are copied directly. Token-row tensors whose first
    dimension is vocab-sized are copied by token name when both vocabularies are
    known, otherwise by overlapping row index. This lets codon-only checkpoints
    initialize hybrid CDS+UTR models while leaving new token rows initialized
    from the target model.
    """
    target_state = model.state_dict()
    adapted = {}
    loaded_exact: list[str] = []
    loaded_rows: list[str] = []
    skipped: list[str] = []

    source_index = {tok: i for i, tok in enumerate(source_itos or [])}
    target_index = {tok: i for i, tok in enumerate(target_itos or [])}

    for name, target_tensor in target_state.items():
        source_tensor = source_state.get(name)
        if source_tensor is None:
            skipped.append(name)
            continue
        if tuple(source_tensor.shape) == tuple(target_tensor.shape):
            adapted[name] = source_tensor
            loaded_exact.append(name)
            continue
        if (
            source_tensor.ndim >= 1
            and target_tensor.ndim >= 1
            and tuple(source_tensor.shape[1:]) == tuple(target_tensor.shape[1:])
            and source_tensor.shape[0] != target_tensor.shape[0]
        ):
            merged = target_tensor.detach().clone()
            copied = 0
            if source_index and target_index:
                for tok, dst_idx in target_index.items():
                    src_idx = source_index.get(tok)
                    if src_idx is None or src_idx >= source_tensor.shape[0] or dst_idx >= merged.shape[0]:
                        continue
                    merged[dst_idx] = source_tensor[src_idx].to(device=merged.device, dtype=merged.dtype)
                    copied += 1
            else:
                copied = min(int(source_tensor.shape[0]), int(merged.shape[0]))
                merged[:copied] = source_tensor[:copied].to(device=merged.device, dtype=merged.dtype)
            if copied:
                adapted[name] = merged
                loaded_rows.append(f"{name}:{copied}")
            else:
                skipped.append(name)
            continue
        skipped.append(name)

    missing, unexpected = model.load_state_dict(adapted, strict=False)
    return {
        "loaded_exact": loaded_exact,
        "loaded_rows": loaded_rows,
        "skipped": skipped,
        "missing": list(missing),
        "unexpected": list(unexpected),
    }


def offset_target_mask(yb: torch.Tensor, offset: int, boundary_ids=DEFAULT_BOUNDARY_IDS) -> torch.Tensor:
    """Return valid positions for predicting seq[t + offset] from logits at t."""
    if offset < 1:
        raise ValueError("offset must be >= 1")
    if offset > yb.shape[1]:
        return torch.zeros((yb.shape[0], 0), dtype=torch.bool, device=yb.device)

    target = yb[:, offset - 1 :]
    valid = target != PAD_ID
    boundary = torch.zeros_like(yb, dtype=torch.bool)
    for boundary_id in boundary_ids:
        boundary |= yb == int(boundary_id)

    # Do not train targets that require looking beyond an EOS/SEP boundary.
    # The target boundary itself is allowed; only earlier boundaries invalidate it.
    for shift in range(offset - 1):
        valid &= ~boundary[:, shift : shift + target.shape[1]]
    return valid


def multi_offset_lm_loss(
    logits: torch.Tensor | dict[int, torch.Tensor],
    yb: torch.Tensor,
    offset_weights: dict[int, float],
    label_smoothing: float = 0.0,
    loss_weights: torch.Tensor | None = None,
    boundary_ids=DEFAULT_BOUNDARY_IDS,
):
    losses = {}
    total = logits.new_tensor(0.0) if hasattr(logits, "new_tensor") else torch.tensor(0.0, device=yb.device)
    for offset, weight in offset_weights.items():
        if weight == 0.0 or offset <= 1 or offset > yb.shape[1]:
            continue
        target = yb[:, offset - 1 :]
        if isinstance(logits, dict):
            if offset not in logits:
                continue
            pred = logits[offset][:, : target.shape[1], :]
        else:
            pred = logits[:, : target.shape[1], :]
        valid = offset_target_mask(yb, offset, boundary_ids=boundary_ids)
        if not bool(valid.any()):
            continue
        pred_flat = pred[valid].float()
        target_flat = target[valid]
        offset_loss = torch.nn.functional.cross_entropy(
            pred_flat,
            target_flat,
            ignore_index=PAD_ID,
            label_smoothing=label_smoothing,
            weight=loss_weights,
        )
        losses[offset] = offset_loss
        total = total + (float(weight) * offset_loss)
    return total, losses


def termination_distance_bucket_labels(
    yb: torch.Tensor,
    stop_ids: tuple[int, ...],
    bucket_edges: tuple[int, ...] = (0, 3, 10, 30),
    ignore_index: int = -100,
) -> torch.Tensor:
    """Bucket distance from each target position to the next stop token.

    Class 0 means the immediate target token is a stop. Later classes represent
    increasingly distant stops; the final class also covers positions with no
    future stop before padding.
    """
    if not stop_ids:
        raise ValueError("stop_ids must not be empty")
    if tuple(bucket_edges) != tuple(sorted(bucket_edges)):
        raise ValueError("bucket_edges must be sorted")

    labels = torch.full_like(yb, fill_value=ignore_index, dtype=torch.long)
    n_classes = len(bucket_edges) + 1
    for row_idx in range(yb.shape[0]):
        row = yb[row_idx]
        valid_positions = row != PAD_ID
        stop_positions = torch.nonzero(
            torch.isin(row, torch.tensor(stop_ids, device=row.device)),
            as_tuple=False,
        ).flatten()
        for pos in torch.nonzero(valid_positions, as_tuple=False).flatten():
            future_stops = stop_positions[stop_positions >= pos]
            if len(future_stops) == 0:
                labels[row_idx, pos] = n_classes - 1
                continue
            distance = int(future_stops[0].item() - pos.item())
            bucket = 0
            while bucket < len(bucket_edges) and distance > int(bucket_edges[bucket]):
                bucket += 1
            labels[row_idx, pos] = bucket
    return labels


def termination_aux_loss(
    termination_logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(
        termination_logits.float().view(-1, termination_logits.size(-1)),
        labels.contiguous().view(-1),
        ignore_index=ignore_index,
    )


def dev(force_gpu: bool = False):
    device = default_device()
    if force_gpu and device.type == "cpu":
        raise RuntimeError("force_gpu=true but no CUDA or MPS device is available")
    return device

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run_id", default=None, help=f"Unique run id; falls back to ${RUN_ID_ENV} or config.run_id")
    ap.add_argument("--resume", default=None, help="Path to checkpoint to resume training from")
    ap.add_argument("--transfer_from", default=None, help="Path to pre-trained weights to initialize model from (ignores optimizer/step state)")
    ap.add_argument("--train_npz", action="append", default=None, help="Training NPZ file (repeatable)")
    ap.add_argument("--val_npz", action="append", default=None, help="Validation NPZ file (repeatable)")
    ap.add_argument("--test_npz", action="append", default=None, help="Test NPZ file (repeatable)")
    ap.add_argument("--save_epochs", action="store_true", help="Save checkpoint at every epoch")
    ap.add_argument("--max_time_minutes", type=float, default=None, help="Override config max_time_minutes")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config)) or {}
    if "data" in cfg and isinstance(cfg["data"], dict):
        for k, v in cfg["data"].items():
            cfg.setdefault(k, v)
    cfg["save_epochs"] = args.save_epochs or cfg.get("save_epochs", False)
    if args.max_time_minutes is not None:
        cfg["max_time_minutes"] = float(args.max_time_minutes)
    resume_path = args.resume or cfg.pop("resume", None)
    if resume_path is not None:
        resume_path = str(resume_path)

    default_train = f"data/processed/train_bs{cfg['block_size']}.npz"
    default_val = f"data/processed/val_bs{cfg['block_size']}.npz"
    default_test = f"data/processed/test_bs{cfg['block_size']}.npz"
    cfg.setdefault("train_npz", default_train)
    cfg.setdefault("val_npz", default_val)
    cfg.setdefault("test_npz", default_test)

    train_paths = _ensure_path_list(args.train_npz, cfg.get("train_npz"), "train_npz")
    val_paths = _ensure_path_list(args.val_npz, cfg.get("val_npz"), "val_npz")
    test_paths = _ensure_path_list(args.test_npz, cfg.get("test_npz"), "test_npz")
    cfg["train_npz"] = train_paths
    cfg["val_npz"] = val_paths
    cfg["test_npz"] = test_paths

    if resume_path and not os.path.isfile(resume_path):
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

    transfer_path = None if resume_path else (args.transfer_from or cfg.pop("transfer_from", None))
    if transfer_path and not os.path.isfile(transfer_path):
        raise FileNotFoundError(f"Transfer weights not found: {transfer_path}")

    matmul_precision = cfg.get("matmul_precision")
    if matmul_precision:
        setter = getattr(torch, "set_float32_matmul_precision", None)
        if callable(setter):
            try:
                setter(str(matmul_precision))
                print(f"[matmul] float32 precision set to {matmul_precision}")
            except Exception as exc:
                print(f"[matmul] failed to set precision '{matmul_precision}': {exc}")
        else:
            print("[matmul] torch.set_float32_matmul_precision unavailable in this build.")

    # Optional: derive embedding size from per-head dimension
    if "d_head" in cfg and cfg.get("n_head"):
        try:
            cfg["n_embd"] = int(cfg["d_head"]) * int(cfg["n_head"])
            print(f"[dims] using d_head={cfg['d_head']} × n_head={cfg['n_head']} → n_embd={cfg['n_embd']}")
        except Exception as exc:
            print(f"[dims] failed to derive n_embd from d_head: {exc}")

    base_seed = int(cfg.get("seed", 1337))
    use_mmap = bool(cfg.get("use_mmap", False))
    train_ds, val_ds = build_codon_lm_datasets(train_paths, val_paths, use_mmap=use_mmap)
    if use_mmap:
        print("[loader] using MmapPackedDataset (memory-mapped, on-demand paging)")
    train_audit = dataset_length_audit(train_ds, int(cfg["block_size"]))
    val_audit = dataset_length_audit(val_ds, int(cfg["block_size"]))
    cfg["dataset_audit"] = {"train": train_audit, "val": val_audit}
    cfg["whole_gene_status"] = (
        "whole-or-truncated"
        if train_audit["at_block_size"] or val_audit["at_block_size"]
        else "whole-under-block-size"
    )
    print(f"[audit] train_lengths={train_audit}")
    print(f"[audit] val_lengths={val_audit}")

    def _loader_cfg_for_epoch(epoch_idx: int) -> dict:
        loader_cfg = dict(cfg)
        loader_cfg["dataloader_seed"] = base_seed + max(0, int(epoch_idx))
        return loader_cfg

    try:
        train_loader, val_loader, train_sampler, dl_kwargs = build_codon_lm_dataloaders(
            train_ds,
            val_ds,
            _loader_cfg_for_epoch(0),
        )
        if train_sampler is not None:
            print(
                f"[loader] BucketBatchSampler: {cfg.get('n_buckets', 8)} buckets, "
                f"{len(train_sampler)} batches, batch_size={cfg['batch_size']}"
            )
    except Exception as exc:
        raise RuntimeError(f"failed to build CodonLM dataloaders: {exc}") from exc

    sep_mask_enabled = bool(cfg.get("sep_mask_enabled", True))
    multi_offset_enabled = bool(cfg.get("multi_offset_loss_enabled", False))
    multi_offset_targets = [int(x) for x in cfg.get("multi_offset_targets", [])]
    multi_offset_weights = (
        _normalize_offset_weights(multi_offset_targets, cfg.get("multi_offset_weights"))
        if multi_offset_enabled
        else {}
    )
    if multi_offset_weights:
        print(f"[loss] multi_offset_weights={multi_offset_weights}")
    termination_loss_enabled = bool(cfg.get("termination_loss_enabled", False))
    termination_loss_weight = float(cfg.get("termination_loss_weight", 0.1))
    termination_stop_ids = tuple(int(x) for x in cfg.get("termination_stop_ids", [2]))
    termination_bucket_edges = tuple(int(x) for x in cfg.get("termination_bucket_edges", [0, 3, 10, 30]))
    termination_n_classes = int(cfg.get("termination_n_classes", len(termination_bucket_edges) + 1))
    if termination_n_classes != len(termination_bucket_edges) + 1:
        raise ValueError("termination_n_classes must equal len(termination_bucket_edges) + 1")
    replay_loss_enabled = bool(cfg.get("replay_loss_enabled", False))
    replay_loss_weight = float(cfg.get("replay_loss_weight", 0.1))
    replay_data = cfg.get("replay_data")
    replay_batch_size = int(cfg.get("replay_batch_size", cfg.get("batch_size", 1)))
    termination_head_enabled = termination_loss_enabled or replay_loss_enabled
    if termination_loss_enabled:
        print(
            f"[loss] termination_aux weight={termination_loss_weight} "
            f"stop_ids={termination_stop_ids} bucket_edges={termination_bucket_edges}"
        )
    if replay_loss_enabled:
        if not replay_data:
            raise ValueError("replay_loss_enabled=true requires replay_data")
        print(
            f"[loss] replay_termination weight={replay_loss_weight} "
            f"data={replay_data} batch_size={replay_batch_size}"
        )

    eos_loss_weight = cfg.get("eos_loss_weight", None)
    loss_weights = None
    if eos_loss_weight is not None and float(eos_loss_weight) != 1.0:
        from .codon_tokenize import stoi, STOP_CODONS
        loss_weights = [1.0] * cfg["vocab_size"]
        loss_weights[stoi["<EOS_CDS>"]] = float(eos_loss_weight)
        for codon in STOP_CODONS:
            if codon in stoi:
                loss_weights[stoi[codon]] = float(eos_loss_weight)
        print(f"[weights] upweighting termination tokens by {eos_loss_weight}x")

    run_id = _normalize_run_id(args.run_id or cfg.get("run_id") or os.environ.get(RUN_ID_ENV))
    if not run_id:
        run_id = _auto_run_id(cfg, args.config)
    if run_id:
        cfg["run_id"] = run_id
    outdir = cfg["out_dir"]
    scores_base = cfg.get("scores_dir", "outputs/scores")
    ckpt_dir, scores_dir = _prepare_output_dirs(outdir, scores_base, run_id)

    shutil.copy2(args.config, ckpt_dir / "config.yaml")
    run_logger = RunLogger(ckpt_dir.parent / "logs" / "train.log")
    run_logger.__enter__()

    def write_failure_meta(exc: Exception) -> None:
        meta = {
            "run_id": run_id,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "model_spec": {},
        }
        write_meta(ckpt_dir, meta)

    log_csv_cfg = cfg.get("log_csv")
    if log_csv_cfg:
        log_csv_path = Path(log_csv_cfg)
        log_csv = (scores_dir / log_csv_path).resolve() if not log_csv_path.is_absolute() else log_csv_path
    else:
        log_csv = scores_dir / "curves.csv"
    log_csv.parent.mkdir(parents=True, exist_ok=True)
    with log_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        offset_cols = []
        for offset in sorted(multi_offset_weights):
            offset_cols.extend([f"train_offset_{offset}", f"val_offset_{offset}"])
        term_cols = ["train_term_loss", "val_term_loss"] if termination_loss_enabled else []
        replay_cols = ["train_replay_term_loss"] if replay_loss_enabled else []
        writer.writerow([
            "step",
            "train_loss",
            "val_loss",
            "train_next_loss",
            "val_next_loss",
            "perplexity",
            "lr",
            *offset_cols,
            *term_cols,
            *replay_cols,
        ])

    try:
        device = dev(force_gpu=bool(cfg.get("force_gpu", False)))
    except Exception as exc:
        print(f"[error] training failed: {exc}", file=sys.stderr)
        write_failure_meta(exc)
        raise

    cfg["device"] = str(device)
    print(f"[device] using {device}")
    torch.manual_seed(base_seed)
    amp = bool(cfg.get("amp", True)) and (device.type == "mps")

    model = TinyGPT(
        cfg["vocab_size"],
        cfg["block_size"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        n_embd=cfg["n_embd"],
        dropout=cfg["dropout"],
        use_checkpoint=bool(cfg.get("use_checkpoint", cfg.get("grad_checkpointing", False))),
        label_smoothing=float(cfg.get("label_smoothing", 0.0)),
        sep_id=(3 if sep_mask_enabled else None),
        tie_embeddings=bool(cfg.get("tie_embeddings", True)),
        n_kv_head=int(cfg.get("n_kv_head")) if cfg.get("n_kv_head") is not None else None,
        use_sdpa=bool(cfg.get("use_sdpa", False)),
        loss_weights=loss_weights,
        termination_aux=termination_head_enabled,
        termination_n_classes=termination_n_classes,
        multi_offset_targets=multi_offset_targets if multi_offset_enabled else None,
        use_swiglu=bool(cfg.get("use_swiglu", False)),
        use_rope=bool(cfg.get("use_rope", False)),
    ).to(device)

    replay_loader = None
    replay_iter = None
    if replay_loss_enabled:
        replay_path = Path(str(replay_data))
        if not replay_path.is_absolute():
            replay_path = Path.cwd() / replay_path
        replay_ds = GeneratedTerminationReplayDataset(
            replay_path,
            block_size=int(cfg["block_size"]),
            pad_id=PAD_ID,
        )
        replay_generator = torch.Generator()
        replay_generator.manual_seed(base_seed + 17)
        replay_loader = DataLoader(
            replay_ds,
            batch_size=max(1, replay_batch_size),
            shuffle=True,
            num_workers=0,
            drop_last=False,
            generator=replay_generator,
        )
        cfg["replay_data"] = str(replay_path)
        cfg["replay_examples"] = int(len(replay_ds))
        print(f"[replay] loaded {len(replay_ds)} generated-state records from {replay_path}")

    compile_requested = bool(cfg.get("compile", False))
    compile_mode = cfg.get("compile_mode", "default")

    if compile_requested:
        # Be resilient: if Dynamo hits a backend quirk, fall back to eager without crashing
        try:
            import torch._dynamo as dynamo
            dynamo.config.suppress_errors = True
            try:
                dynamo.config.log_level = logging.ERROR
            except Exception:
                pass
        except Exception:
            pass
        # Compatibility shim for older transformers packages that break Dynamo's ModelOutput probe
        try:
            import importlib
            fu = importlib.import_module("transformers.file_utils")
            if not hasattr(fu, "ModelOutput"):
                utils_mod = importlib.import_module("transformers.utils")
                if hasattr(utils_mod, "ModelOutput"):
                    setattr(fu, "ModelOutput", getattr(utils_mod, "ModelOutput"))
        except Exception:
            # transformers not installed or different layout; ignore
            pass
        torch_compile = getattr(torch, "compile", None)
        if torch_compile:
            try:
                model = torch_compile(model, mode=compile_mode)
                print(f"[compile] torch.compile enabled (mode={compile_mode})")
                # Probe whether a graph was actually captured; if not, we likely fell back to eager.
                try:
                    from torch._dynamo.utils import counters as _dynamo_counters  # type: ignore
                    before_ok = int(_dynamo_counters["frames"].get("ok", 0)) if isinstance(_dynamo_counters, dict) else 0
                    probe_T = max(1, min(8, int(cfg.get("block_size", 8))))
                    with torch.no_grad():
                        _ = model(torch.zeros((1, probe_T), dtype=torch.long, device=device))
                    after_ok = int(_dynamo_counters["frames"].get("ok", 0)) if isinstance(_dynamo_counters, dict) else before_ok
                    captured = max(0, after_ok - before_ok)
                    if captured == 0:
                        print("[compile] no graphs captured; running in eager (fallback).")
                    else:
                        print(f"[compile] graphs_captured={captured}")
                except Exception:
                    # If counters are unavailable or the probe fails, stay silent
                    pass
            except Exception as exc:
                print(f"[compile] torch.compile failed ({exc}); continuing without compilation.")
        else:
            print("[compile] torch.compile not available in this PyTorch build.")

    # Backbone freezing
    freeze_backbone = bool(cfg.get("freeze_backbone", False))
    if freeze_backbone:
        frozen_count = 0
        trainable_count = 0
        for name, param in model.named_parameters():
            if "offset_projs" in name or "termination_head" in name:
                param.requires_grad = True
                trainable_count += 1
            else:
                param.requires_grad = False
                frozen_count += 1
        print(f"[freeze] Backbone frozen: {frozen_count} tensors frozen, {trainable_count} tensors trainable (offset_projs and termination_head)")

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    # Optimizer selection (lazy import to avoid importing transformers unless needed)
    if cfg.get("optimizer", "adamw").lower() == "adafactor":
        try:
            from transformers.optimization import Adafactor  # type: ignore
        except Exception:
            raise RuntimeError("transformers not installed; pip install transformers to use Adafactor")
        optim = Adafactor(
            trainable_params,
            lr=cfg.get("lr", 3e-4),
            scale_parameter=False,
            relative_step=False,
            weight_decay=cfg.get("weight_decay", 0.05),
        )
    else:
        optim = torch.optim.AdamW(trainable_params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    scheduler_name = str(cfg.get("scheduler", "cosine")).lower()
    if scheduler_name not in {"cosine", "plateau"}:
        print(f"[warn] Unknown scheduler '{scheduler_name}', defaulting to cosine.")
        scheduler_name = "cosine"

    gacc = cfg.get("grad_accum_steps", 16)
    warmup_steps = int(cfg.get("warmup_steps", 200))
    min_lr = float(cfg.get("min_lr", 1e-5))
    base_lr = float(cfg["lr"])

    # Auto-epochs: target total tokens = tokens_per_param * n_params (heuristic T = 20 * N)
    # tokens_per_epoch ≈ len(train_ds) * block_size
    epochs_cfg = cfg.get("epochs", 5)
    n_params = sum(p.numel() for p in model.parameters())
    tokens_per_param = float(cfg.get("tokens_per_param", 20.0))
    if isinstance(epochs_cfg, str) and epochs_cfg.strip().lower() == "auto":
        tokens_target = max(1.0, tokens_per_param * float(n_params))
        tokens_per_epoch = max(1.0, float(len(train_ds) * cfg["block_size"]))
        est_epochs = int(math.ceil(tokens_target / tokens_per_epoch))
        # clamp to reasonable bounds
        est_epochs = max(int(cfg.get("epochs_min", 1)), min(est_epochs, int(cfg.get("epochs_max", max(1, est_epochs)))))
        max_epochs = est_epochs
        print(
            f"[epochs-auto] tokens_per_param={tokens_per_param} n_params={n_params} → target_tokens={int(tokens_target)}; "
            f"tokens_per_epoch≈{int(tokens_per_epoch)} → epochs={max_epochs}"
        )
    else:
        max_epochs = int(epochs_cfg)
    steps_per_epoch = math.ceil(len(train_loader) / max(1, gacc))
    total_steps = max(1, steps_per_epoch * max_epochs)
    use_cosine = scheduler_name == "cosine"
    if use_cosine:
        warmup_for_lambda = max(1, warmup_steps)
        min_lr_ratio = (min_lr / base_lr) if base_lr > 0 else 0.0

        def lr_lambda(step_idx: int) -> float:
            if step_idx < warmup_for_lambda:
                return float(step_idx + 1) / warmup_for_lambda
            progress = (step_idx - warmup_for_lambda) / max(1, total_steps - warmup_for_lambda)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1 - min_lr_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim,
            mode="min",
            factor=0.5,
            patience=cfg.get("plateau_patience", 2),
            min_lr=min_lr,
        )

    start_epoch = 0
    best = float("inf")
    no_improve = 0
    step = 0
    best_epoch = None
    resume_microbatch_idx = 0
    current_epoch_idx = 0
    current_microbatch_idx = 0
    current_resume_microbatch_idx = 0

    try:
        if transfer_path:
            print(f"[transfer] initializing model from {transfer_path}")
            ckpt_transfer = torch.load(transfer_path, map_location=device)
            sd = ckpt_transfer["model"] if isinstance(ckpt_transfer, dict) and "model" in ckpt_transfer else ckpt_transfer
            transfer_cfg = ckpt_transfer.get("cfg", {}) if isinstance(ckpt_transfer, dict) else {}
            source_itos = _read_itos(transfer_cfg.get("itos_path"), Path.cwd())
            target_itos = _read_itos(cfg.get("itos_path"), Path.cwd())
            transfer_report = _load_transfer_state_dict(
                model,
                sd,
                source_itos=source_itos,
                target_itos=target_itos,
            )
            print(
                "[transfer] loaded_exact="
                f"{len(transfer_report['loaded_exact'])} row_loaded={transfer_report['loaded_rows']}"
            )
            if transfer_report["skipped"]:
                print(f"[transfer] skipped_shape_or_missing={transfer_report['skipped']}")
            if transfer_report["missing"]:
                print(f"[transfer] missing_after_adapt={transfer_report['missing']}")
            if transfer_report["unexpected"]:
                print(f"[transfer] unexpected_after_adapt={transfer_report['unexpected']}")

        if resume_path:
            print(f"[resume] loading {resume_path}")
            ckpt_resume = torch.load(resume_path, map_location=device)
            model.load_state_dict(ckpt_resume["model"])
            if "optimizer" in ckpt_resume:
                try:
                    optim.load_state_dict(ckpt_resume["optimizer"])
                except Exception as exc:
                    print(f"[resume] optimizer state load failed: {exc}")
            if "scheduler" in ckpt_resume and ckpt_resume["scheduler"] is not None:
                try:
                    scheduler.load_state_dict(ckpt_resume["scheduler"])
                except Exception as exc:
                    print(f"[resume] scheduler state load failed: {exc}")
            start_epoch = int(ckpt_resume.get("epoch", 0))
            step = int(ckpt_resume.get("step", step))
            best = float(ckpt_resume.get("best_val", best))
            best_epoch = ckpt_resume.get("best_epoch", best_epoch)
            no_improve = int(ckpt_resume.get("no_improve", no_improve))
            resume_microbatch_idx = int(ckpt_resume.get("epoch_microbatch_idx", 0) or 0)
            checkpoint_batch_size = ckpt_resume.get("batch_size")
            checkpoint_grad_accum = ckpt_resume.get("grad_accum_steps")
            if checkpoint_batch_size is not None and int(checkpoint_batch_size) != int(cfg["batch_size"]):
                print(
                    "[resume] warning: checkpoint batch_size="
                    f"{checkpoint_batch_size} but current batch_size={cfg['batch_size']}; "
                    "mid-epoch resume position will be ignored."
                )
                resume_microbatch_idx = 0
            if checkpoint_grad_accum is not None and int(checkpoint_grad_accum) != int(gacc):
                print(
                    "[resume] warning: checkpoint grad_accum_steps="
                    f"{checkpoint_grad_accum} but current grad_accum_steps={gacc}; "
                    "mid-epoch resume position will be ignored."
                )
                resume_microbatch_idx = 0
            if resume_microbatch_idx:
                print(f"[resume] will skip {resume_microbatch_idx} completed microbatches in epoch {start_epoch + 1}")

        if start_epoch >= max_epochs:
            print(f"[resume] start_epoch {start_epoch} >= configured epochs {max_epochs}; no new epochs will run unless you increase 'epochs'.")

        periodic_ckpt = PeriodicCheckpointPolicy(
            every_steps=int(cfg.get("checkpoint_every_steps", 0) or 0),
            every_minutes=float(cfg.get("checkpoint_every_minutes", 0.0) or 0.0),
            last_saved_step=step,
        )

        def make_checkpoint_payload(
            epoch_idx: int,
            train_loss: float = float("inf"),
            val_loss: float = float("inf"),
            train_next_loss: float | None = None,
            val_next_loss: float | None = None,
            train_term_loss: float | None = None,
            val_term_loss: float | None = None,
            train_replay_term_loss: float | None = None,
        ) -> dict:
            return {
                "model": model.state_dict(),
                "optimizer": optim.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "cfg": cfg,
                "epoch": max(0, epoch_idx - 1) if val_loss == float("inf") else epoch_idx,
                "val_loss": val_loss,
                "train_loss": train_loss,
                "train_next_loss": train_next_loss,
                "val_next_loss": val_next_loss,
                "train_term_loss": train_term_loss,
                "val_term_loss": val_term_loss,
                "train_replay_term_loss": train_replay_term_loss,
                "best_val": best,
                "best_epoch": best_epoch,
                "no_improve": no_improve,
                "step": step,
                "epoch_microbatch_idx": (
                    0 if val_loss != float("inf") else int(current_resume_microbatch_idx)
                ),
                "last_seen_microbatch_idx": int(current_microbatch_idx),
                "batch_size": int(cfg["batch_size"]),
                "grad_accum_steps": int(gacc),
                "train_examples": int(len(train_ds)),
                "train_batches": int(len(train_loader)),
            }

        def save_last_checkpoint(epoch_idx: int, reason: str, **metrics) -> None:
            payload = make_checkpoint_payload(epoch_idx, **metrics)
            payload["checkpoint_reason"] = reason
            save_checkpoint_atomic(payload, ckpt_dir / "last.pt")
            periodic_ckpt.mark_saved(step)
            print(f"[checkpoint] saved {ckpt_dir / 'last.pt'} reason={reason} step={step}")

        def one_pass(split, loader, epoch_idx: int, skip_microbatches: int = 0):
            """Runs a single training or validation epoch pass over the dataset."""
            nonlocal step, current_epoch_idx, current_microbatch_idx, current_resume_microbatch_idx, replay_iter
            mps_autocast_ok = True
            model.train(split=="train")
            total, next_total, term_total, replay_total, n = 0.0, 0.0, 0.0, 0.0, 0
            term_count = 0
            replay_count = 0
            offset_totals = {offset: 0.0 for offset in multi_offset_weights}
            offset_counts = {offset: 0 for offset in multi_offset_weights}
            optim.zero_grad(set_to_none=True)
            skipped = 0
            start_time = time.perf_counter()
            if split == "train" and skip_microbatches > 0:
                print(f"[resume] skipping {skip_microbatches}/{len(loader)} already-applied train microbatches")
            for batch_idx, (xb, yb) in enumerate(loader):
                current_epoch_idx = epoch_idx
                if split == "train" and batch_idx < skip_microbatches:
                    current_microbatch_idx = batch_idx + 1
                    current_resume_microbatch_idx = batch_idx + 1
                    continue
                current_microbatch_idx = batch_idx + 1
                if n > 0 and n % 200 == 0:
                    elapsed = time.perf_counter() - start_time
                    seen = batch_idx if split == "train" else n
                    print(f"[{split}] progress: {seen}/{len(loader)} speed: {n*xb.shape[0]/elapsed:.2f} seq/sec")

                xb, yb = xb.to(device), yb.to(device)
                def fwd():
                    """Helper to run model forward pass and return scaled loss."""
                    nonlocal replay_iter
                    need_aux = termination_loss_enabled or bool(multi_offset_weights)
                    if need_aux:
                        logits_, next_loss_, aux_ = model(xb, yb, return_aux=True)
                    else:
                        logits_, next_loss_ = model(xb, yb)
                        aux_ = {}
                    total_loss_ = next_loss_
                    offset_losses_ = {}
                    if multi_offset_weights:
                        offset_logits_input = aux_.get("offset_logits", logits_)
                        offset_total_, offset_losses_ = multi_offset_lm_loss(
                            offset_logits_input,
                            yb,
                            multi_offset_weights,
                            label_smoothing=float(cfg.get("label_smoothing", 0.0)),
                            loss_weights=(
                                model.loss_weights
                                if not torch.all(model.loss_weights == 1.0).item()
                                else None
                            ),
                        )
                        total_loss_ = total_loss_ + offset_total_
                    term_loss_ = None
                    if termination_loss_enabled:
                        term_logits = aux_.get("termination_logits")
                        if term_logits is None:
                            raise RuntimeError("termination_loss_enabled=true but model returned no termination logits")
                        term_labels = termination_distance_bucket_labels(
                            yb,
                            stop_ids=termination_stop_ids,
                            bucket_edges=termination_bucket_edges,
                        )
                        term_loss_ = termination_aux_loss(term_logits, term_labels)
                        total_loss_ = total_loss_ + (termination_loss_weight * term_loss_)
                    replay_loss_ = None
                    if split == "train" and replay_loader is not None:
                        if replay_iter is None:
                            replay_iter = iter(replay_loader)
                        try:
                            replay_x, replay_labels = next(replay_iter)
                        except StopIteration:
                            replay_iter = iter(replay_loader)
                            replay_x, replay_labels = next(replay_iter)
                        replay_x = replay_x.to(device)
                        replay_labels = replay_labels.to(device)
                        _, _, replay_aux = model(replay_x, return_aux=True)
                        replay_logits = replay_aux.get("termination_logits")
                        if replay_logits is None:
                            raise RuntimeError("replay_loss_enabled=true but model returned no termination logits")
                        replay_loss_ = termination_aux_loss(replay_logits, replay_labels)
                        total_loss_ = total_loss_ + (replay_loss_weight * replay_loss_)
                    return total_loss_ / gacc, next_loss_, offset_losses_, term_loss_, replay_loss_

                if amp and mps_autocast_ok:
                    try:
                        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=True):
                            loss, next_loss, offset_losses, term_loss, replay_loss = fwd()
                    except RuntimeError as e:
                        msg = str(e).lower()
                        if "unsupported autocast device_type" in msg or "autocast" in msg and device.type == "mps":
                            # Fallback: disable MPS autocast for this run
                            mps_autocast_ok = False
                            loss, next_loss, offset_losses, term_loss, replay_loss = fwd()
                        else:
                            raise
                else:
                    loss, next_loss, offset_losses, term_loss, replay_loss = fwd()
                # Guard against NaNs/infs (e.g., degenerate batches); skip update for this micro-step
                if not torch.isfinite(loss):
                    skipped += 1
                    continue
                if split=="train":
                    loss.backward()
                    if (n+1) % gacc == 0:
                        if (not use_cosine) and warmup_steps > 0 and step < warmup_steps:
                            scale = float(step + 1) / max(1, warmup_steps)
                            for pg in optim.param_groups:
                                pg["lr"] = base_lr * scale
                        optim.step()
                        optim.zero_grad(set_to_none=True)
                        step += 1
                        current_resume_microbatch_idx = batch_idx + 1
                        if use_cosine:
                            scheduler.step()
                        if device.type == "mps":
                            torch.mps.empty_cache()
                        if split == "train" and periodic_ckpt.should_save(step):
                            save_last_checkpoint(epoch_idx, reason="periodic")
                            if device.type == "mps":
                                torch.mps.empty_cache()
                total += loss.item()*gacc
                next_total += float(next_loss.detach().item())
                if term_loss is not None:
                    term_total += float(term_loss.detach().item())
                    term_count += 1
                if replay_loss is not None:
                    replay_total += float(replay_loss.detach().item())
                    replay_count += 1
                for offset, offset_loss in offset_losses.items():
                    offset_totals[offset] += float(offset_loss.detach().item())
                    offset_counts[offset] += 1
                n += 1
                wall_timer.check()
            offset_avgs = {
                offset: (offset_totals[offset] / max(offset_counts[offset], 1))
                for offset in offset_totals
            }
            return (
                total / max(n, 1),
                next_total / max(n, 1),
                (term_total / max(term_count, 1)) if termination_loss_enabled else None,
                (replay_total / max(replay_count, 1)) if replay_loss_enabled and split == "train" else None,
                skipped,
                offset_avgs,
            )

        history = []

        if run_id:
            print(f"[run] id={run_id}")
        print(f"[paths] ckpts={ckpt_dir} scores={scores_dir} log_csv={log_csv}")
        print(f"[model] params={n_params} sep_mask_enabled={sep_mask_enabled}")
        print(
            f"[loader] num_workers={dl_kwargs.get('num_workers')} "
            f"pin_memory={dl_kwargs.get('pin_memory')} "
            f"prefetch_factor={dl_kwargs.get('prefetch_factor')} "
            f"persistent_workers={dl_kwargs.get('persistent_workers', False)}"
        )
        print(
            f"[train] starting: epochs={max_epochs}, steps_per_epoch={steps_per_epoch}, total_steps={total_steps}, "
            f"batch_size={cfg['batch_size']}, grad_accum={gacc}, scheduler={scheduler_name}"
        )

        train_wall0 = time.perf_counter()
        train_cpu0 = time.process_time()

        max_time_minutes = cfg.get("max_time_minutes", None)
        if max_time_minutes:
            print(f"[*] Wall-time limit configured: {max_time_minutes} minutes")
        wall_timer = WallTimer(max_time_minutes)

        for epoch in range(start_epoch, max_epochs):
            train_loader, _, train_sampler, _ = build_codon_lm_dataloaders(
                train_ds,
                val_ds,
                _loader_cfg_for_epoch(epoch + 1),
            )
            ep_wall0 = time.perf_counter()
            ep_cpu0 = time.process_time()
            epoch_idx = epoch + 1
            skip_for_epoch = resume_microbatch_idx if epoch == start_epoch else 0
            resume_microbatch_idx = 0
            train_loss, train_next_loss, train_term_loss, train_replay_term_loss, train_skips, train_offsets = one_pass(
                "train",
                train_loader,
                epoch_idx,
                skip_microbatches=skip_for_epoch,
            )
            with torch.no_grad():
                val_loss, val_next_loss, val_term_loss, _, val_skips, val_offsets = one_pass("val", val_loader, epoch_idx)
            ppl = math.exp(min(20.0, val_next_loss))
            if not use_cosine:
                scheduler.step(val_loss)
            lr_now = optim.param_groups[0]["lr"]
            msg = (
                f"[epoch {epoch_idx}] train {train_loss:.3f} | val {val_loss:.3f} "
                f"| next_val {val_next_loss:.3f} | ppl {ppl:.2f} | lr {lr_now:.2e}"
            )
            if train_skips or val_skips:
                msg += f" | skips train={train_skips} val={val_skips}"
            if multi_offset_weights:
                offset_msg = " ".join(
                    f"o{offset}:train={train_offsets.get(offset, 0.0):.3f}/val={val_offsets.get(offset, 0.0):.3f}"
                    for offset in sorted(multi_offset_weights)
                )
                msg += f" | offsets {offset_msg}"
            if termination_loss_enabled:
                msg += f" | term train={train_term_loss:.3f}/val={val_term_loss:.3f}"
            if replay_loss_enabled:
                msg += f" | replay_term train={train_replay_term_loss:.3f}"
            print(msg)
            ep_wall1 = time.perf_counter()
            ep_cpu1 = time.process_time()
            print(f"[timing] epoch {epoch_idx} wall_sec={ep_wall1-ep_wall0:.2f} cpu_sec={ep_cpu1-ep_cpu0:.2f}")

            improved = val_loss + 1e-6 < best
            if improved:
                best = val_loss
                best_epoch = epoch_idx
                no_improve = 0
            else:
                no_improve += 1

            ckpt_payload = make_checkpoint_payload(
                epoch_idx,
                train_loss=train_loss,
                val_loss=val_loss,
                train_next_loss=train_next_loss,
                val_next_loss=val_next_loss,
                train_term_loss=train_term_loss,
                val_term_loss=val_term_loss,
                train_replay_term_loss=train_replay_term_loss,
            )
            save_checkpoint_atomic(ckpt_payload, ckpt_dir / "last.pt")
            periodic_ckpt.mark_saved(step)
            if cfg.get("save_epochs", False):
                save_checkpoint_atomic(ckpt_payload, ckpt_dir / f"epoch_{epoch_idx}.pt")
            with log_csv.open("a", newline="") as f:
                row = [
                    epoch_idx,
                    f"{train_loss:.4f}",
                    f"{val_loss:.4f}",
                    f"{train_next_loss:.4f}",
                    f"{val_next_loss:.4f}",
                    f"{ppl:.3f}",
                    f"{lr_now:.3e}",
                ]
                for offset in sorted(multi_offset_weights):
                    row.extend([
                        f"{train_offsets.get(offset, 0.0):.4f}",
                        f"{val_offsets.get(offset, 0.0):.4f}",
                    ])
                if termination_loss_enabled:
                    row.extend([
                        f"{train_term_loss:.4f}",
                        f"{val_term_loss:.4f}",
                    ])
                if replay_loss_enabled:
                    row.append(f"{train_replay_term_loss:.4f}")
                csv.writer(f).writerow(row)

            history.append({
                "epoch": epoch_idx,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_next_loss": train_next_loss,
                "val_next_loss": val_next_loss,
                "train_term_loss": train_term_loss,
                "val_term_loss": val_term_loss,
                "train_replay_term_loss": train_replay_term_loss,
                "perplexity": ppl,
                "lr": lr_now,
            })

            if improved:
                save_checkpoint_atomic(ckpt_payload, ckpt_dir / "best.pt")
            elif no_improve >= int(cfg.get("early_stop_patience", 5)):
                print("[early-stopping] no improvement; stopping.")
                break
    except WallTimeLimitException:
        print(f"\n[info] Wall-time limit of {max_time_minutes} minutes reached mid-epoch.")
        ckpt_payload = make_checkpoint_payload(current_epoch_idx or (start_epoch + 1))
        ckpt_payload["checkpoint_reason"] = "wall_time"
        save_checkpoint_atomic(ckpt_payload, ckpt_dir / "last.pt")
        print(f"[success] Gracefully saved checkpoint to {ckpt_dir / 'last.pt'}. Exiting.")

        # Final meta and metrics collection
        train_wall1 = time.perf_counter()
        train_cpu1 = time.process_time()
        total_time = train_wall1 - train_wall0

        meta = {
            "run_id": run_id,
            "train_wall_sec": round(total_time, 2),
            "train_cpu_sec": round(train_cpu1 - train_cpu0, 2),
            "best_epoch": best_epoch,
            "best_val_loss": float(best) if best != float("inf") else None,
            "status": "stopped",
            "model_spec": model.to_dict() if hasattr(model, "to_dict") else {}
        }

        if history:
            meta.update({
                "last_epoch": history[-1]["epoch"],
                "last_val_loss": history[-1]["val_loss"],
                "last_train_loss": history[-1]["train_loss"],
                "last_val_next_loss": history[-1].get("val_next_loss"),
                "last_train_next_loss": history[-1].get("train_next_loss"),
                "last_val_term_loss": history[-1].get("val_term_loss"),
                "last_train_term_loss": history[-1].get("train_term_loss"),
                "last_train_replay_term_loss": history[-1].get("train_replay_term_loss"),
                "last_perplexity": history[-1]["perplexity"],
            })
            metrics_path = scores_dir / "metrics.json"
            metrics_path.write_text(json.dumps(meta, indent=2) + "\n")

        write_meta(ckpt_dir, meta)

        print(f"[timing] train_wall_sec={total_time:.2f} train_cpu_sec={train_cpu1-train_cpu0:.2f}")
        return
    except Exception as exc:
        print(f"[error] training failed: {exc}", file=sys.stderr)
        write_failure_meta(exc)
        raise

    # Final meta and metrics collection
    train_wall1 = time.perf_counter()
    train_cpu1 = time.process_time()
    total_time = train_wall1 - train_wall0

    meta = {
        "run_id": run_id,
        "train_wall_sec": round(total_time, 2),
        "train_cpu_sec": round(train_cpu1 - train_cpu0, 2),
        "best_epoch": best_epoch,
        "best_val_loss": float(best) if best != float("inf") else None,
        "status": "completed",
        "model_spec": model.to_dict() if hasattr(model, "to_dict") else {}
    }

    if history:
        meta.update({
            "last_epoch": history[-1]["epoch"],
            "last_val_loss": history[-1]["val_loss"],
            "last_train_loss": history[-1]["train_loss"],
            "last_val_next_loss": history[-1].get("val_next_loss"),
            "last_train_next_loss": history[-1].get("train_next_loss"),
            "last_val_term_loss": history[-1].get("val_term_loss"),
            "last_train_term_loss": history[-1].get("train_term_loss"),
            "last_train_replay_term_loss": history[-1].get("train_replay_term_loss"),
            "last_perplexity": history[-1]["perplexity"],
        })

        # Also write the compact metrics.json for dashboard
        metrics_path = scores_dir / "metrics.json"
        metrics_path.write_text(json.dumps(meta, indent=2) + "\n")

    write_meta(ckpt_dir, meta)

    print(f"[timing] train_wall_sec={total_time:.2f} train_cpu_sec={train_cpu1-train_cpu0:.2f}")

if __name__ == "__main__":
    main()
