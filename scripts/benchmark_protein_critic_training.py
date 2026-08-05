"""Subprocess-isolated ProteinCritic MPS throughput benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml

from src.protein_lm.config import ProteinClassifierConfig
from src.protein_lm.dataset import MultiTaskProteinDataset, collate_protein_batch
from src.protein_lm.models_multi import MultiTaskProteinClassifier
from src.protein_lm.tokenizer import ProteinTokenizer
from src.protein_lm.train_multi_task import task_losses


def stratified_indices(dataset, count: int) -> list[int]:
    """Select a deterministic length-stratified sample, including both endpoints."""
    count = min(int(count), len(dataset))
    if count < 1:
        raise ValueError("sample count must be positive")
    ordered = sorted(range(len(dataset)), key=dataset.sequence_length)
    if count == 1:
        return [ordered[len(ordered) // 2]]
    positions = [round(i * (len(ordered) - 1) / (count - 1)) for i in range(count)]
    return [ordered[position] for position in positions]


def candidate_batches(indices: list[int], batch_size: int) -> list[list[int]]:
    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    return [indices[i : i + batch_size] for i in range(0, len(indices), batch_size)]


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _memory(device: torch.device) -> tuple[int, int]:
    if device.type == "mps":
        current = getattr(torch.mps, "current_allocated_memory", lambda: 0)()
        driver = getattr(torch.mps, "driver_allocated_memory", lambda: 0)()
        return int(current), int(driver)
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device)), int(
            torch.cuda.max_memory_reserved(device)
        )
    return 0, 0


def _device(name: str) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def run_worker(args) -> dict:
    cfg = yaml.safe_load(Path(args.config).read_text()) or {}
    device = _device(args.device)
    if args.force_gpu and device.type == "cpu":
        raise RuntimeError(
            "GPU benchmark requested but neither MPS nor CUDA is available"
        )
    torch.manual_seed(int(cfg.get("seed", 1337)))

    tokenizer = ProteinTokenizer()
    vocabs = json.loads(Path(cfg["task_vocabs"]).read_text())
    regression_tasks = tuple(cfg.get("regression_tasks", []))
    classification_tasks = tuple(
        task
        for task in ("family", "function", "stability")
        if task not in regression_tasks
    )
    task_dims = {
        "family": len(vocabs["pfam"]),
        "function": len(vocabs["ec"]),
        "stability": 1 if "stability" in regression_tasks else len(vocabs["stability"]),
    }
    block_size = int(args.block_size or cfg.get("block_size", 512))
    model_cfg = ProteinClassifierConfig(
        vocab_size=len(tokenizer.vocab),
        block_size=block_size,
        n_layer=int(cfg.get("n_layer", 4)),
        n_head=int(cfg.get("n_head", 4)),
        n_embd=int(cfg.get("n_embd", 128)),
        dropout=float(cfg.get("dropout", 0.1)),
        num_classes=0,
        use_checkpoint=bool(args.use_checkpoint),
        pooling=cfg.get("pooling", "mean"),
        bidirectional=bool(cfg.get("bidirectional", True)),
    )
    model = MultiTaskProteinClassifier(model_cfg, task_dims).to(device).train()
    dataset = MultiTaskProteinDataset(
        cfg["train_data"],
        tokenizer,
        max_length=block_size,
        dynamic_padding=True,
    )
    effective_batch = int(args.batch_size) * int(args.grad_accum_steps)
    measured_indices = stratified_indices(dataset, effective_batch * int(args.groups))
    batches = candidate_batches(measured_indices, args.batch_size)
    warmup_indices = stratified_indices(dataset, args.batch_size)
    warmup_batches = candidate_batches(warmup_indices, args.batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 1e-4)))
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

    def execute(
        batch_indices: list[int], divisor: int
    ) -> tuple[int, int, dict[str, float]]:
        batch = collate_protein_batch(
            [dataset[index] for index in batch_indices], tokenizer.pad_token_id
        )
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits = model(input_ids, attention_mask=attention_mask)
        losses = task_losses(
            logits,
            {
                task: batch[task].to(device)
                for task in (*classification_tasks, *regression_tasks)
            },
            classification_tasks,
            regression_tasks,
            criterion,
        )
        if not losses:
            raise RuntimeError("benchmark batch has no supervised targets")
        (torch.stack(list(losses.values())).mean() / divisor).backward()
        task_values = {
            name: float(value.detach().cpu()) for name, value in losses.items()
        }
        return int(attention_mask.sum()), int(input_ids.numel()), task_values

    optimizer.zero_grad(set_to_none=True)
    for batch_indices in warmup_batches:
        execute(batch_indices, len(warmup_batches))
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    _sync(device)

    if device.type == "mps":
        reset_peak = getattr(torch.mps, "reset_peak_memory_stats", None)
        if callable(reset_peak):
            reset_peak()
    elif device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    batch_seconds = []
    optimizer_seconds = []
    useful_residues = 0
    padded_tokens = 0
    task_sums: dict[str, float] = {}
    task_counts: dict[str, int] = {}
    peak_current = 0
    peak_driver = 0
    started = time.perf_counter()
    for group in range(int(args.groups)):
        group_batches = batches[
            group * int(args.grad_accum_steps) : (group + 1)
            * int(args.grad_accum_steps)
        ]
        optimizer.zero_grad(set_to_none=True)
        for batch_indices in group_batches:
            _sync(device)
            before = time.perf_counter()
            useful, padded, values = execute(batch_indices, len(group_batches))
            _sync(device)
            batch_seconds.append(time.perf_counter() - before)
            useful_residues += useful
            padded_tokens += padded
            for task, value in values.items():
                task_sums[task] = task_sums.get(task, 0.0) + value
                task_counts[task] = task_counts.get(task, 0) + 1
            current, driver = _memory(device)
            peak_current = max(peak_current, current)
            peak_driver = max(peak_driver, driver)
        _sync(device)
        before = time.perf_counter()
        optimizer.step()
        _sync(device)
        optimizer_seconds.append(time.perf_counter() - before)
    wall_seconds = time.perf_counter() - started

    ordered = sorted(batch_seconds)

    def percentile(fraction: float) -> float:
        index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
        return ordered[index]

    seq_count = len(measured_indices)
    seq_per_sec = seq_count / wall_seconds
    return {
        "status": "ok",
        "device": str(device),
        "batch_size": int(args.batch_size),
        "grad_accum_steps": int(args.grad_accum_steps),
        "effective_batch_size": effective_batch,
        "block_size": block_size,
        "use_checkpoint": bool(args.use_checkpoint),
        "groups": int(args.groups),
        "sequences": seq_count,
        "useful_residues": useful_residues,
        "padded_tokens": padded_tokens,
        "padding_fraction": 1.0 - useful_residues / max(padded_tokens, 1),
        "wall_seconds": wall_seconds,
        "seq_per_sec": seq_per_sec,
        "residues_per_sec": useful_residues / wall_seconds,
        "estimated_train_hours_per_epoch": len(dataset) / seq_per_sec / 3600.0,
        "microbatch_p50_seconds": percentile(0.50),
        "microbatch_p95_seconds": percentile(0.95),
        "microbatch_max_seconds": max(batch_seconds),
        "optimizer_mean_seconds": sum(optimizer_seconds) / len(optimizer_seconds),
        "peak_allocated_bytes": peak_current,
        "peak_driver_bytes": peak_driver,
        "task_losses": {
            task: task_sums[task] / task_counts[task] for task in sorted(task_sums)
        },
        "sample_min_length": min(dataset.sequence_length(i) for i in measured_indices),
        "sample_max_length": max(dataset.sequence_length(i) for i in measured_indices),
    }


def run_candidate(
    config: str, candidate: dict, timeout_seconds: int, force_gpu: bool
) -> dict:
    command = [
        sys.executable,
        "-m",
        "scripts.benchmark_protein_critic_training",
        "--worker",
        "--config",
        config,
        "--batch-size",
        str(candidate["batch_size"]),
        "--grad-accum-steps",
        str(candidate["grad_accum_steps"]),
        "--groups",
        str(candidate.get("groups", 1)),
        "--device",
        str(candidate.get("device", "auto")),
    ]
    if candidate.get("block_size"):
        command.extend(["--block-size", str(candidate["block_size"])])
    if candidate.get("use_checkpoint"):
        command.append("--use-checkpoint")
    if force_gpu:
        command.append("--force-gpu")
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {**candidate, "status": "timeout", "timeout_seconds": timeout_seconds}
    if completed.returncode != 0:
        return {
            **candidate,
            "status": "error",
            "returncode": completed.returncode,
            "error": completed.stderr[-4000:],
        }
    lines = [
        line
        for line in completed.stdout.splitlines()
        if line.startswith("RESULT_JSON=")
    ]
    if not lines:
        return {**candidate, "status": "error", "error": "worker emitted no result"}
    result = json.loads(lines[-1].split("=", 1)[1])
    result["name"] = candidate["name"]
    return result


def write_results(out_dir: Path, results: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps({"results": results}, indent=2) + "\n"
    )
    fields = [
        "name",
        "status",
        "device",
        "batch_size",
        "grad_accum_steps",
        "effective_batch_size",
        "block_size",
        "seq_per_sec",
        "residues_per_sec",
        "estimated_train_hours_per_epoch",
        "padding_fraction",
        "microbatch_p50_seconds",
        "microbatch_p95_seconds",
        "microbatch_max_seconds",
        "optimizer_mean_seconds",
        "peak_allocated_bytes",
        "peak_driver_bytes",
        "error",
    ]
    with (out_dir / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--matrix")
    parser.add_argument("--out", default="runs/protein_critic_mps_benchmark")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--force-gpu", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--grad-accum-steps", type=int)
    parser.add_argument("--groups", type=int, default=1)
    parser.add_argument("--block-size", type=int)
    parser.add_argument("--use-checkpoint", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    if args.worker:
        result = run_worker(args)
        print("RESULT_JSON=" + json.dumps(result, sort_keys=True), flush=True)
        return
    if not args.matrix:
        raise ValueError("--matrix is required for a benchmark sweep")
    matrix = yaml.safe_load(Path(args.matrix).read_text()) or {}
    results = []
    for candidate in matrix.get("candidates", []):
        print(f"[critic-bench] running {candidate['name']}", flush=True)
        result = run_candidate(
            args.config,
            candidate,
            timeout_seconds=int(args.timeout_seconds),
            force_gpu=bool(args.force_gpu),
        )
        results.append(result)
        if result["status"] == "ok":
            print(
                f"[critic-bench] {candidate['name']}: "
                f"{result['seq_per_sec']:.2f} seq/s, "
                f"{result['estimated_train_hours_per_epoch']:.2f} train h/epoch, "
                f"driver={result['peak_driver_bytes'] / 1024**3:.2f} GiB",
                flush=True,
            )
        else:
            print(f"[critic-bench] {candidate['name']}: {result['status']}", flush=True)
        write_results(Path(args.out), results)


if __name__ == "__main__":
    main()
