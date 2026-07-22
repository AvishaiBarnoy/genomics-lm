#!/usr/bin/env python3
"""Benchmark and optionally apply CodonLM batch/grad-accum settings."""

from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import yaml

from src.codonlm.data_loading import build_codon_lm_dataloaders, build_codon_lm_datasets
from src.codonlm.model_tiny_gpt import TinyGPT
from src.codonlm.training.checkpoint import _load_transfer_state_dict, _read_itos
from src.codonlm.training.loop import _average_accumulated_gradients
from src.codonlm.training.objectives import (
    termination_aux_loss,
    termination_distance_bucket_labels,
)
from src.training.runtime import default_device


DEFAULT_CANDIDATES = [(2, 16), (4, 16), (4, 32), (8, 16), (8, 32)]
OOM_PATTERNS = (
    "out of memory",
    "mps backend out of memory",
    "mps allocated",
    "cuda out of memory",
    "allocation",
    "failed to allocate",
)


def _process_max_rss_bytes() -> int:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw if sys.platform == "darwin" else raw * 1024


def load_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text()) or {}


def apply_data_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cfg = dict(cfg)
    for arg_name, cfg_name in (
        ("train_npz", "train_npz"),
        ("val_npz", "val_npz"),
        ("test_npz", "test_npz"),
    ):
        value = getattr(args, arg_name, None)
        if value is not None:
            cfg[cfg_name] = [str(path) for path in value]
    return cfg


def parse_candidates(value: str | None) -> list[tuple[int, int]]:
    if not value:
        return list(DEFAULT_CANDIDATES)
    out: list[tuple[int, int]] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "/" in item:
            bsz, gacc = item.split("/", 1)
        elif ":" in item:
            bsz, gacc = item.split(":", 1)
        else:
            raise ValueError(f"Bad candidate {item!r}; use B/GACC")
        out.append((int(bsz), int(gacc)))
    if not out:
        raise ValueError("No candidates parsed")
    return out


def candidates_from_config(raw: Any) -> list[tuple[int, int]] | None:
    if raw is None:
        return None
    candidates: list[tuple[int, int]] = []
    for item in raw:
        if isinstance(item, str):
            candidates.extend(parse_candidates(item))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            candidates.append((int(item[0]), int(item[1])))
        elif isinstance(item, dict):
            candidates.append((int(item["batch_size"]), int(item["grad_accum_steps"])))
        else:
            raise ValueError(f"Bad candidate spec: {item!r}")
    return candidates


def dedupe_candidates(candidates: list[tuple[int, int]]) -> list[tuple[int, int]]:
    seen = set()
    out: list[tuple[int, int]] = []
    for batch_size, grad_accum_steps in candidates:
        pair = (int(batch_size), int(grad_accum_steps))
        if pair in seen:
            continue
        seen.add(pair)
        out.append(pair)
    return out


def include_current_candidate(
    cfg: dict[str, Any],
    candidates: list[tuple[int, int]],
    include_current: bool,
) -> list[tuple[int, int]]:
    if not include_current:
        return dedupe_candidates(candidates)
    if "batch_size" not in cfg or "grad_accum_steps" not in cfg:
        return dedupe_candidates(candidates)
    current = (int(cfg["batch_size"]), int(cfg["grad_accum_steps"]))
    return dedupe_candidates([current, *candidates])


def resolve_optimizer_settings(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    section = cfg.get("batch_optimizer") or {}
    cli_mode = args.mode
    if getattr(args, "benchmark", False):
        cli_mode = "benchmark"
    if getattr(args, "optimize", False):
        cli_mode = "optimize"
    mode = cli_mode or section.get("mode") or "benchmark"
    if mode not in {"benchmark", "optimize"}:
        raise ValueError("mode must be 'benchmark' or 'optimize'")
    candidates = (
        parse_candidates(args.candidates)
        if args.candidates
        else candidates_from_config(section.get("candidates")) or list(DEFAULT_CANDIDATES)
    )
    include_current = bool(section.get("include_current", True))
    candidates = include_current_candidate(cfg, candidates, include_current)
    return {
        "mode": mode,
        "candidates": candidates,
        "include_current": include_current,
        "warmup_steps": int(args.warmup_steps if args.warmup_steps is not None else section.get("warmup_steps", 20)),
        "measure_steps": int(args.measure_steps if args.measure_steps is not None else section.get("measure_steps", 100)),
        "force_gpu": bool(args.force_gpu or section.get("force_gpu", cfg.get("force_gpu", False))),
        "force": bool(getattr(args, "force", False) or section.get("force", False)),
        "include_in_wall_time": bool(section.get("include_in_wall_time", True)),
        "min_training_minutes_after_opt": float(section.get("min_training_minutes_after_opt", 0.0) or 0.0),
    }


def select_device(force_gpu: bool) -> torch.device:
    device = default_device()
    if force_gpu and device.type == "cpu":
        raise RuntimeError("force_gpu=true but neither CUDA nor MPS is available")
    return device


def _path_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, os.PathLike)):
        return [str(value)]
    return [str(v) for v in value]


def build_model(cfg: dict[str, Any], device: torch.device) -> TinyGPT:
    model = TinyGPT(
        int(cfg["vocab_size"]),
        int(cfg["block_size"]),
        n_layer=int(cfg["n_layer"]),
        n_head=int(cfg["n_head"]),
        n_embd=int(cfg["n_embd"]),
        dropout=float(cfg.get("dropout", 0.0)),
        use_checkpoint=bool(cfg.get("use_checkpoint", cfg.get("grad_checkpointing", False))),
        label_smoothing=float(cfg.get("label_smoothing", 0.0)),
        sep_id=(3 if bool(cfg.get("sep_mask_enabled", True)) else None),
        tie_embeddings=bool(cfg.get("tie_embeddings", True)),
        n_kv_head=int(cfg.get("n_kv_head")) if cfg.get("n_kv_head") is not None else None,
        use_sdpa=bool(cfg.get("use_sdpa", False)),
        termination_aux=bool(cfg.get("termination_loss_enabled", cfg.get("termination_aux", False))),
        termination_n_classes=int(cfg.get("termination_n_classes", 5)),
        multi_offset_targets=(
            [int(value) for value in cfg.get("multi_offset_targets", [])]
            if bool(cfg.get("multi_offset_loss_enabled", False))
            else None
        ),
        use_swiglu=bool(cfg.get("use_swiglu", False)),
        use_rope=bool(cfg.get("use_rope", False)),
        use_shape_guidance=bool(cfg.get("use_shape_guidance", False)),
    ).to(device)
    transfer_from = cfg.get("transfer_from")
    if transfer_from:
        ckpt = torch.load(str(transfer_from), map_location=device)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        source_cfg = ckpt.get("cfg", {}) if isinstance(ckpt, dict) else {}
        _load_transfer_state_dict(
            model,
            state,
            source_itos=_read_itos(source_cfg.get("itos_path"), Path.cwd()),
            target_itos=_read_itos(cfg.get("itos_path"), Path.cwd()),
        )
    if bool(cfg.get("freeze_backbone", False)):
        for name, param in model.named_parameters():
            param.requires_grad = "offset_projs" in name or "termination_head" in name
    model.train(True)
    return model


def run_candidate_benchmark(
    cfg: dict[str, Any],
    *,
    batch_size: int,
    grad_accum_steps: int,
    warmup_steps: int,
    measure_steps: int,
    force_gpu: bool,
    device_override: torch.device | None = None,
) -> dict[str, Any]:
    cfg = dict(cfg)
    cfg["batch_size"] = int(batch_size)
    cfg["grad_accum_steps"] = int(grad_accum_steps)
    device = device_override or select_device(force_gpu)
    seed = int(cfg.get("seed", 1337))
    torch.manual_seed(seed)
    cfg.setdefault("dataloader_seed", seed)

    train_paths = _path_list(cfg.get("train_npz", cfg.get("train_paths")))
    if not train_paths:
        raise ValueError("Config must define train_npz or train_paths")
    val_paths = _path_list(cfg.get("val_npz", cfg.get("val_paths"))) or train_paths
    rss_start_bytes = _process_max_rss_bytes()
    train_ds, val_ds = build_codon_lm_datasets(
        train_paths, val_paths, use_mmap=bool(cfg.get("use_mmap", False))
    )
    rss_after_dataset_bytes = _process_max_rss_bytes()
    loader, _, _, _ = build_codon_lm_dataloaders(train_ds, val_ds, cfg)
    loader_iter = iter(loader)

    model = build_model(cfg, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("lr", 3e-4)),
        weight_decay=float(cfg.get("weight_decay", 0.05)),
    )

    termination_enabled = bool(cfg.get("termination_loss_enabled", False))
    termination_weight = float(cfg.get("termination_loss_weight", 0.0))
    stop_ids = tuple(int(x) for x in cfg.get("termination_stop_ids", [2]))
    bucket_edges = tuple(int(x) for x in cfg.get("termination_bucket_edges", [0, 3, 10, 30]))

    amp = bool(cfg.get("amp", True)) and device.type == "mps"
    mps_autocast_ok = True
    parameters = [param for param in model.parameters() if param.requires_grad]

    def synchronize() -> None:
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()

    def forward_loss(xb: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
        if termination_enabled:
            _, next_loss, aux = model(xb, yb, return_aux=True)
            term_labels = termination_distance_bucket_labels(yb, stop_ids=stop_ids, bucket_edges=bucket_edges)
            term_loss = termination_aux_loss(aux["termination_logits"], term_labels)
            return next_loss + (termination_weight * term_loss)
        _, loss = model(xb, yb)
        return loss

    def run_phase(microbatches: int, *, measure: bool) -> dict[str, Any]:
        nonlocal loader_iter, mps_autocast_ok
        stats: dict[str, Any] = {
            "sequences": 0,
            "processed_tokens": 0,
            "non_pad_tokens": 0,
            "times": [],
            "optimizer_steps": 0,
            "peak_allocated_bytes": 0,
            "peak_driver_bytes": 0,
        }
        accumulated = 0
        optimizer.zero_grad(set_to_none=True)
        for idx in range(max(0, int(microbatches))):
            try:
                xb, yb = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                xb, yb = next(loader_iter)
            xb = xb.to(device)
            yb = yb.to(device)
            synchronize()
            t0 = time.perf_counter()
            if amp and mps_autocast_ok:
                try:
                    with torch.amp.autocast(device_type="mps", dtype=torch.float16):
                        loss = forward_loss(xb, yb)
                except RuntimeError as exc:
                    if "autocast" not in str(exc).lower():
                        raise
                    mps_autocast_ok = False
                    loss = forward_loss(xb, yb)
            else:
                loss = forward_loss(xb, yb)
            loss.backward()
            accumulated += 1
            if measure and device.type == "mps":
                stats["peak_allocated_bytes"] = max(
                    stats["peak_allocated_bytes"], int(torch.mps.current_allocated_memory())
                )
                driver_memory = getattr(torch.mps, "driver_allocated_memory", None)
                if callable(driver_memory):
                    stats["peak_driver_bytes"] = max(
                        stats["peak_driver_bytes"], int(driver_memory())
                    )
            is_last = idx + 1 == int(microbatches)
            if accumulated == max(1, int(grad_accum_steps)) or is_last:
                _average_accumulated_gradients(parameters, accumulated)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accumulated = 0
                stats["optimizer_steps"] += 1
            synchronize()
            elapsed = time.perf_counter() - t0
            if measure:
                stats["times"].append(elapsed)
                stats["sequences"] += int(xb.shape[0])
                stats["processed_tokens"] += int(yb.numel())
                stats["non_pad_tokens"] += int(yb.ne(0).sum().item())
                if device.type == "cuda":
                    stats["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        return stats

    run_phase(int(warmup_steps), measure=False)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    measured_stats = run_phase(int(measure_steps), measure=True)

    if device.type == "mps":
        torch.mps.empty_cache()
    times = measured_stats["times"]
    measured = max(sum(times), 1e-9)
    total_sequences = int(measured_stats["sequences"])
    total_tokens = int(measured_stats["processed_tokens"])
    non_pad_tokens = int(measured_stats["non_pad_tokens"])
    optimizer_steps = int(measured_stats["optimizer_steps"])
    seq_per_sec = total_sequences / measured
    tokens_per_sec = total_tokens / measured
    non_pad_tokens_per_sec = non_pad_tokens / measured
    train_len = len(train_ds)
    process_peak_rss_bytes = _process_max_rss_bytes()
    return {
        "status": "ok",
        "device": str(device),
        "batch_size": int(batch_size),
        "grad_accum_steps": int(grad_accum_steps),
        "effective_batch_size": int(batch_size) * int(grad_accum_steps),
        "warmup_steps": int(warmup_steps),
        "measure_steps": int(measure_steps),
        "seq_per_sec": seq_per_sec,
        "tokens_per_sec": tokens_per_sec,
        "non_pad_tokens_per_sec": non_pad_tokens_per_sec,
        "processed_tokens": total_tokens,
        "non_pad_tokens": non_pad_tokens,
        "padding_fraction": 1.0 - (non_pad_tokens / max(total_tokens, 1)),
        "optimizer_steps": optimizer_steps,
        "avg_microbatch_ms": (sum(times) / max(len(times), 1)) * 1000.0,
        "wall_sec_per_optimizer_step": measured / max(optimizer_steps, 1),
        "peak_allocated_bytes": int(measured_stats["peak_allocated_bytes"]),
        "peak_driver_bytes": int(measured_stats["peak_driver_bytes"]),
        "process_rss_start_bytes": rss_start_bytes,
        "process_rss_after_dataset_bytes": rss_after_dataset_bytes,
        "dataset_rss_delta_bytes": max(0, rss_after_dataset_bytes - rss_start_bytes),
        "process_peak_rss_bytes": process_peak_rss_bytes,
        "amp_requested": bool(cfg.get("amp", True)),
        "amp_active": amp and mps_autocast_ok,
        "avg_step_ms": (sum(times) / max(len(times), 1)) * 1000.0,
        "train_windows": int(train_len),
        "estimated_epoch_sec": train_len / max(seq_per_sec, 1e-9),
    }


def classify_failure(text: str) -> str:
    lowered = text.lower()
    if any(pattern in lowered for pattern in OOM_PATTERNS):
        return "oom_or_allocation"
    return "runtime_error"


def select_best_result(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    passed = [r for r in results if r.get("status") == "ok"]
    if not passed:
        return None
    return sorted(
        passed,
        key=lambda r: (
            -float(r.get("non_pad_tokens_per_sec", r.get("tokens_per_sec", r.get("seq_per_sec", 0.0)))),
            int(r["batch_size"]),
            int(r["grad_accum_steps"]),
        ),
    )[0]


def output_dir_for(run_id: str) -> Path:
    return Path("runs") / run_id / "scores" / "batch_optimizer"


def benchmark_signature(config_path: Path, cfg: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    train_paths = _path_list(cfg.get("train_npz", cfg.get("train_paths")))
    relevant = {
        "config_path": str(config_path),
        "train_paths": train_paths,
        "vocab_size": cfg.get("vocab_size"),
        "block_size": cfg.get("block_size"),
        "n_layer": cfg.get("n_layer"),
        "n_head": cfg.get("n_head"),
        "n_kv_head": cfg.get("n_kv_head"),
        "n_embd": cfg.get("n_embd"),
        "dropout": cfg.get("dropout"),
        "use_checkpoint": cfg.get("use_checkpoint", cfg.get("grad_checkpointing", False)),
        "freeze_backbone": cfg.get("freeze_backbone", False),
        "use_sdpa": cfg.get("use_sdpa", False),
        "sep_mask_enabled": cfg.get("sep_mask_enabled", True),
        "use_mmap": cfg.get("use_mmap", False),
        "bucket_batching": cfg.get("bucket_batching", False),
        "n_buckets": cfg.get("n_buckets"),
        "num_workers": cfg.get("num_workers", 0),
        "pin_memory": cfg.get("pin_memory", False),
        "prefetch_factor": cfg.get("prefetch_factor"),
        "persistent_workers": cfg.get("persistent_workers"),
        "optimizer": cfg.get("optimizer", "adamw"),
        "termination_loss_enabled": cfg.get("termination_loss_enabled", False),
        "termination_n_classes": cfg.get("termination_n_classes", 5),
        "transfer_from": cfg.get("transfer_from"),
        "candidates": [list(candidate) for candidate in settings["candidates"]],
        "include_current": bool(settings.get("include_current", True)),
        "warmup_steps": int(settings["warmup_steps"]),
        "measure_steps": int(settings["measure_steps"]),
        "force_gpu": bool(settings["force_gpu"]),
        "include_in_wall_time": bool(settings.get("include_in_wall_time", True)),
        "min_training_minutes_after_opt": float(settings.get("min_training_minutes_after_opt", 0.0)),
    }
    return relevant


def apply_remaining_wall_time_budget(
    cfg: dict[str, Any],
    *,
    elapsed_seconds: float,
    include_in_wall_time: bool,
) -> tuple[dict[str, Any], float | None]:
    selected_cfg = dict(cfg)
    max_time_minutes = selected_cfg.get("max_time_minutes")
    if not include_in_wall_time or max_time_minutes is None:
        return selected_cfg, None
    remaining_minutes = float(max_time_minutes) - (float(elapsed_seconds) / 60.0)
    selected_cfg["max_time_minutes"] = max(0.0, remaining_minutes)
    selected_cfg["batch_optimizer_elapsed_minutes"] = float(elapsed_seconds) / 60.0
    selected_cfg["batch_optimizer_original_max_time_minutes"] = float(max_time_minutes)
    return selected_cfg, remaining_minutes


def benchmark_signatures_compatible(cached: dict[str, Any], current: dict[str, Any]) -> bool:
    if cached == current:
        return True
    comparable_cached = {k: v for k, v in cached.items() if k != "config_sha256"}
    for key, value in comparable_cached.items():
        if current.get(key) != value:
            return False
    return True


def load_cached_report(out_dir: Path, signature: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any] | None] | None:
    results_path = out_dir / "results.json"
    meta_path = out_dir / "benchmark_meta.json"
    if not (results_path.exists() and meta_path.exists()):
        return None
    try:
        meta = json.loads(meta_path.read_text())
        payload = json.loads(results_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    cached_signature = meta.get("signature")
    if not isinstance(cached_signature, dict) or not benchmark_signatures_compatible(cached_signature, signature):
        return None
    results = payload.get("results")
    if not isinstance(results, list):
        return None
    return results, payload.get("selected")


def write_report(
    out_dir: Path,
    results: list[dict[str, Any]],
    selected: dict[str, Any] | None,
    selected_cfg: dict[str, Any],
    train_command: list[str],
    signature: dict[str, Any] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "status",
        "failure_kind",
        "device",
        "batch_size",
        "grad_accum_steps",
        "effective_batch_size",
        "seq_per_sec",
        "tokens_per_sec",
        "non_pad_tokens_per_sec",
        "padding_fraction",
        "optimizer_steps",
        "avg_microbatch_ms",
        "wall_sec_per_optimizer_step",
        "peak_allocated_bytes",
        "peak_driver_bytes",
        "amp_active",
        "avg_step_ms",
        "estimated_epoch_sec",
        "returncode",
    ]
    with (out_dir / "results.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    (out_dir / "results.json").write_text(
        json.dumps({"results": results, "selected": selected}, indent=2, sort_keys=True) + "\n"
    )
    if signature is not None:
        (out_dir / "benchmark_meta.json").write_text(
            json.dumps({"signature": signature, "written_at": time.time()}, indent=2, sort_keys=True) + "\n"
        )
    (out_dir / "selected_config.yaml").write_text(yaml.safe_dump(selected_cfg, sort_keys=False))
    (out_dir / "train_command.txt").write_text(" ".join(train_command) + "\n")


def train_command_for(config_path: Path, run_id: str, resume: str | None = None) -> list[str]:
    cmd = [sys.executable, "-m", "src.codonlm.train_codon_lm", "--config", str(config_path), "--run_id", run_id]
    if resume:
        cmd.extend(["--resume", resume])
    return cmd


def run_child(payload_path: Path) -> int:
    payload = json.loads(payload_path.read_text())
    try:
        result = run_candidate_benchmark(**payload)
    except Exception as exc:
        result = {
            "status": "failed",
            "failure_kind": classify_failure(str(exc)),
            "error": str(exc),
            "traceback": traceback.format_exc(limit=12),
            "batch_size": payload.get("batch_size"),
            "grad_accum_steps": payload.get("grad_accum_steps"),
        }
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result.get("status") == "ok" else 2


def run_candidate_subprocess(
    cfg: dict[str, Any],
    candidate: tuple[int, int],
    *,
    warmup_steps: int,
    measure_steps: int,
    force_gpu: bool,
) -> dict[str, Any]:
    payload = {
        "cfg": cfg,
        "batch_size": candidate[0],
        "grad_accum_steps": candidate[1],
        "warmup_steps": warmup_steps,
        "measure_steps": measure_steps,
        "force_gpu": force_gpu,
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(payload, fh)
        payload_path = Path(fh.name)
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "scripts.optimize_train_batching", "--_candidate_payload", str(payload_path)],
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
        )
    finally:
        payload_path.unlink(missing_ok=True)

    result = None
    for line in reversed((proc.stdout or "").splitlines()):
        try:
            result = json.loads(line)
            break
        except json.JSONDecodeError:
            continue
    if result is None:
        result = {
            "status": "failed",
            "batch_size": candidate[0],
            "grad_accum_steps": candidate[1],
            "failure_kind": classify_failure((proc.stdout or "") + "\n" + (proc.stderr or "")),
            "error": "candidate subprocess did not return JSON",
        }
    result["returncode"] = proc.returncode
    if proc.stdout:
        result["stdout_tail"] = proc.stdout[-2000:]
    if proc.stderr:
        result["stderr_tail"] = proc.stderr[-2000:]
    return result


def main(argv: list[str] | None = None) -> int:
    suite_started_at = time.perf_counter()
    ap = argparse.ArgumentParser(description="Optimize CodonLM batch_size/grad_accum_steps")
    ap.add_argument("--config")
    ap.add_argument("--run_id")
    ap.add_argument("--resume")
    ap.add_argument("--train_npz", action="append", default=None, help="Training NPZ file override (repeatable)")
    ap.add_argument("--val_npz", action="append", default=None, help="Validation NPZ file override (repeatable)")
    ap.add_argument("--test_npz", action="append", default=None, help="Test NPZ file override (repeatable)")
    ap.add_argument("--mode", choices=["benchmark", "optimize"], default=None)
    mode_group = ap.add_mutually_exclusive_group()
    mode_group.add_argument("--benchmark", action="store_true", help="Benchmark candidates and write a report")
    mode_group.add_argument("--optimize", action="store_true", help="Benchmark candidates, then start training")
    ap.add_argument("--candidates", default=None, help="Comma list like 2/16,4/16,8/32")
    ap.add_argument("--warmup_steps", type=int, default=None)
    ap.add_argument("--measure_steps", type=int, default=None)
    ap.add_argument("--force_gpu", action="store_true")
    ap.add_argument("--force", action="store_true", help="Ignore cached benchmark results and rerun candidates")
    ap.add_argument("--_candidate_payload", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args._candidate_payload:
        return run_child(Path(args._candidate_payload))
    if not args.config:
        ap.error("--config is required")

    config_path = Path(args.config)
    cfg = apply_data_overrides(load_yaml(config_path), args)
    settings = resolve_optimizer_settings(cfg, args)
    run_id = args.run_id or cfg.get("run_id")
    if not run_id:
        raise SystemExit("--run_id is required unless config defines run_id")

    select_device(bool(settings["force_gpu"]))
    out_dir = output_dir_for(str(run_id))
    signature = benchmark_signature(config_path, cfg, settings)
    cached = None if settings["force"] else load_cached_report(out_dir, signature)
    if cached is not None:
        results, selected = cached
        print(f"[batch-opt] using cached benchmark results from {out_dir} (set batch_optimizer.force=true or --force to rerun)", flush=True)
    else:
        results = []
        for candidate in settings["candidates"]:
            print(f"[batch-opt] benchmarking batch_size={candidate[0]} grad_accum_steps={candidate[1]}", flush=True)
            result = run_candidate_subprocess(
                cfg,
                candidate,
                warmup_steps=int(settings["warmup_steps"]),
                measure_steps=int(settings["measure_steps"]),
                force_gpu=bool(settings["force_gpu"]),
            )
            status = result.get("status")
            if status == "ok":
                print(
                    f"[batch-opt] ok {candidate[0]}/{candidate[1]} "
                    f"non-pad tok/sec={float(result['non_pad_tokens_per_sec']):.2f} "
                    f"epoch_h={float(result['estimated_epoch_sec']) / 3600.0:.2f}",
                    flush=True,
                )
            else:
                print(f"[batch-opt] failed {candidate[0]}/{candidate[1]} {result.get('failure_kind')}: {result.get('error')}", flush=True)
            results.append(result)
        selected = select_best_result(results)
    elapsed_seconds = time.perf_counter() - suite_started_at
    if selected is None:
        selected_cfg, _ = apply_remaining_wall_time_budget(
            cfg,
            elapsed_seconds=elapsed_seconds,
            include_in_wall_time=bool(settings.get("include_in_wall_time", True)),
        )
        command = []
        write_report(out_dir, results, selected, selected_cfg, command, signature=signature)
        raise SystemExit("[batch-opt] no passing candidates")

    selected_cfg, remaining_minutes = apply_remaining_wall_time_budget(
        cfg,
        elapsed_seconds=elapsed_seconds,
        include_in_wall_time=bool(settings.get("include_in_wall_time", True)),
    )
    selected_cfg["batch_size"] = int(selected["batch_size"])
    selected_cfg["grad_accum_steps"] = int(selected["grad_accum_steps"])
    selected_cfg["force_gpu"] = bool(settings["force_gpu"] or selected_cfg.get("force_gpu", False))
    selected_config_path = out_dir / "selected_config.yaml"
    command = train_command_for(selected_config_path, str(run_id), resume=args.resume)
    write_report(out_dir, results, selected, selected_cfg, command, signature=signature)
    print(
        f"[batch-opt] selected batch_size={selected['batch_size']} "
        f"grad_accum_steps={selected['grad_accum_steps']} "
        "non-pad tok/sec="
        f"{float(selected.get('non_pad_tokens_per_sec', selected.get('tokens_per_sec', selected.get('seq_per_sec', 0.0)))):.2f}",
        flush=True,
    )
    print(f"[batch-opt] wrote report to {out_dir}", flush=True)
    if remaining_minutes is not None:
        original_minutes = float(cfg["max_time_minutes"])
        print(
            f"[batch-opt] end-to-end wall budget: original={original_minutes:.2f}m "
            f"optimizer_elapsed={elapsed_seconds / 60.0:.2f}m "
            f"training_remaining={remaining_minutes:.2f}m",
            flush=True,
        )

    if settings["mode"] == "optimize":
        min_training_minutes = float(settings.get("min_training_minutes_after_opt", 0.0))
        if remaining_minutes is not None and remaining_minutes <= 0:
            raise SystemExit("[batch-opt] wall-time budget exhausted during batch optimization; training not started")
        if remaining_minutes is not None and remaining_minutes < min_training_minutes:
            raise SystemExit(
                "[batch-opt] remaining training budget "
                f"{remaining_minutes:.2f}m is below min_training_minutes_after_opt={min_training_minutes:.2f}m; "
                "training not started"
            )
        print("[batch-opt] starting training with selected_config.yaml", flush=True)
        return subprocess.call(command, cwd=Path.cwd())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
