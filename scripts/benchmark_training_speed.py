"""Subprocess-isolated CodonLM training throughput benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Optional

import torch
import yaml

from scripts.optimize_train_batching import run_candidate_benchmark, run_candidate_subprocess


def _load_cfg(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text()) or {}


def _result_for_config(
    cfg: dict[str, Any],
    *,
    name: str,
    n_steps: int,
    warmup_steps: int,
    device: Optional[torch.device] = None,
) -> dict[str, Any]:
    batch_size = int(cfg["batch_size"])
    grad_accum_steps = int(cfg.get("grad_accum_steps", 1))
    if device is None:
        result = run_candidate_subprocess(
            cfg,
            (batch_size, grad_accum_steps),
            warmup_steps=warmup_steps,
            measure_steps=n_steps,
            force_gpu=bool(cfg.get("force_gpu", False)),
        )
    else:
        result = run_candidate_benchmark(
            cfg,
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
            warmup_steps=warmup_steps,
            measure_steps=n_steps,
            force_gpu=device.type != "cpu",
            device_override=device,
        )
    result["config"] = name
    result["use_checkpoint"] = bool(cfg.get("use_checkpoint", cfg.get("grad_checkpointing", False)))
    result["n_kv_head"] = cfg.get("n_kv_head")
    result["sep_mask_enabled"] = bool(cfg.get("sep_mask_enabled", True))
    result["amp_requested"] = bool(cfg.get("amp", True))
    return result


def benchmark(
    config_path: str,
    n_steps: int = 30,
    warmup_steps: int = 5,
    device: Optional[torch.device] = None,
) -> dict[str, Any]:
    cfg = _load_cfg(config_path)
    return _result_for_config(
        cfg,
        name=Path(config_path).stem,
        n_steps=n_steps,
        warmup_steps=warmup_steps,
        device=device,
    )


def load_matrix(path: str | Path) -> tuple[list[tuple[str, dict[str, Any]]], int, int]:
    manifest = _load_cfg(path)
    base_path = Path(manifest["base_config"])
    if not base_path.is_absolute():
        base_path = Path.cwd() / base_path
    base_cfg = _load_cfg(base_path)
    base_cfg.update(manifest.get("base_overrides", {}))
    variants = []
    for variant in manifest.get("variants", []):
        cfg = dict(base_cfg)
        cfg.update(variant.get("overrides", {}))
        variants.append((str(variant["name"]), cfg))
    if not variants:
        raise ValueError("benchmark matrix must define at least one variant")
    return (
        variants,
        int(manifest.get("warmup_steps", 20)),
        int(manifest.get("measure_steps", 100)),
    )


def write_results(path: str | Path, results: list[dict[str, Any]]) -> None:
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps({"results": results}, indent=2, sort_keys=True) + "\n")
    fields = [
        "config",
        "status",
        "failure_kind",
        "device",
        "batch_size",
        "grad_accum_steps",
        "effective_batch_size",
        "use_checkpoint",
        "n_kv_head",
        "amp_requested",
        "amp_active",
        "non_pad_tokens_per_sec",
        "tokens_per_sec",
        "padding_fraction",
        "seq_per_sec",
        "avg_microbatch_ms",
        "wall_sec_per_optimizer_step",
        "peak_allocated_bytes",
        "peak_driver_bytes",
        "process_rss_start_bytes",
        "process_rss_after_dataset_bytes",
        "dataset_rss_delta_bytes",
        "process_peak_rss_bytes",
        "error",
    ]
    with (out_dir / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark CodonLM training throughput")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--config", nargs="+", help="Training config(s) to benchmark")
    source.add_argument("--matrix", help="YAML manifest containing a base config and named overrides")
    parser.add_argument("--steps", type=int, default=None, help="Measured microbatches")
    parser.add_argument("--warmup", type=int, default=None, help="Warmup microbatches")
    parser.add_argument("--out", default="runs/training_speed_benchmark", help="CSV/JSON output directory")
    args = parser.parse_args(argv)

    if args.matrix:
        variants, default_warmup, default_steps = load_matrix(args.matrix)
    else:
        paths = args.config or [
            "configs/stage2.6_large_scaling.yaml",
            "configs/stage2.6_optimized.yaml",
        ]
        variants = [(Path(path).stem, _load_cfg(path)) for path in paths]
        default_warmup, default_steps = 5, 30
    warmup = int(args.warmup if args.warmup is not None else default_warmup)
    steps = int(args.steps if args.steps is not None else default_steps)

    results = []
    for name, cfg in variants:
        print(f"[bench] running {name}", flush=True)
        result = _result_for_config(cfg, name=name, n_steps=steps, warmup_steps=warmup)
        results.append(result)
        if result.get("status") == "ok":
            print(
                f"[bench] {name}: {float(result['non_pad_tokens_per_sec']):,.0f} non-pad tok/s, "
                f"padding={float(result['padding_fraction']):.1%}",
                flush=True,
            )
        else:
            print(f"[bench] {name}: {result.get('failure_kind', 'error')} {result.get('error', '')}")
    write_results(args.out, results)
    print(f"[bench] wrote {args.out}")


if __name__ == "__main__":
    main()
