#!/usr/bin/env python3
"""Diagnose whether a CodonLM checkpoint uses context beyond token composition."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from scripts.eval_ppl_baselines import (
    _probability,
    evaluate_baselines,
    fit_baselines,
)
from src.codonlm.checkpoints import build_codon_model_from_cfg, load_codon_checkpoint
from src.codonlm.data_loading import MmapPackedDataset
from src.codonlm.evaluation_provenance import (
    artifact_provenance,
    bind_checkpoint_dataset,
    bind_dataset_manifest,
)
from src.codonlm.training.vocabulary import resolve_vocabulary_contract

PAD_ID = 0


def _device(name: str) -> torch.device:
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but unavailable")
        return torch.device("mps")
    return torch.device("cpu")


def _parse_windows(value: str) -> list[int | None]:
    windows: list[int | None] = []
    for raw in value.split(","):
        item = raw.strip().lower()
        window = None if item == "full" else int(item)
        if window is not None and window < 1:
            raise ValueError("context windows must be positive or 'full'")
        if window not in windows:
            windows.append(window)
    if not windows:
        raise ValueError("at least one context window is required")
    return windows


def _evaluation_artifact_role(split: str) -> str:
    if split == "test":
        return "test_tokens"
    if split == "validation":
        return "val_tokens"
    raise ValueError(f"unsupported evaluation split: {split}")


def _packing_window_flags(path: Path | None, n_windows: int) -> np.ndarray:
    flags = np.zeros(n_windows, dtype=bool)
    if path is None:
        return flags
    with path.open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            index = int(row["window_index"])
            if index < 0 or index >= n_windows:
                raise ValueError(f"packing metadata window index out of range: {index}")
            flags[index] |= row["continues_from_previous"] == "1"
            flags[index] |= row["continues_to_next"] == "1"
    return flags


def _trigram_nll(
    x: np.ndarray,
    y: np.ndarray,
    trigram: dict,
    bigram: dict,
    *,
    alpha: float,
    active_size: int,
    reset_token_ids: frozenset[int],
) -> np.ndarray:
    result = np.zeros(len(y), dtype=np.float64)
    for position, target_value in enumerate(y):
        target = int(target_value)
        if target == PAD_ID:
            continue
        previous = int(x[position])
        previous2 = (
            PAD_ID
            if position == 0 or previous in reset_token_ids
            else int(x[position - 1])
        )
        counts = trigram.get((previous2, previous))
        if counts is None:
            counts = bigram.get(previous)
        result[position] = -math.log(
            _probability(counts, target, alpha, active_size)
        )
    return result


def _position_bin(position: int) -> str:
    if position == 0:
        return "segment_position_0"
    if position < 4:
        return "segment_position_1_3"
    if position < 16:
        return "segment_position_4_15"
    if position < 64:
        return "segment_position_16_63"
    return "segment_position_64_plus"


def _token_class(token: str) -> str:
    if token.startswith("<"):
        return "special"
    if token == "ATG":
        return "start_codon"
    if token in {"TAA", "TAG", "TGA"}:
        return "stop_codon"
    return "ordinary_codon"


def _add_slice(
    slices: dict[str, list[float]], name: str, losses: np.ndarray, mask: np.ndarray
) -> None:
    selected = losses[mask]
    if selected.size:
        slices[name][0] += float(selected.sum())
        slices[name][1] += int(selected.size)


def _bootstrap_paired_rows(
    model_rows: np.ndarray,
    baseline_rows: np.ndarray,
    row_tokens: np.ndarray,
    *,
    seed: int,
    samples: int,
) -> dict:
    valid = row_tokens > 0
    differences = model_rows[valid] - baseline_rows[valid]
    tokens = row_tokens[valid]
    observed = float(differences.sum() / tokens.sum())
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for sample in range(samples):
        indices = rng.integers(0, len(tokens), size=len(tokens))
        estimates[sample] = differences[indices].sum() / tokens[indices].sum()
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {
        "codonlm_minus_trigram_nats_per_token": observed,
        "ci95": [float(low), float(high)],
        "bootstrap_unit": "packed_window",
        "bootstrap_samples": samples,
        "seed": seed,
    }


def _mask_audit(model, dataset: MmapPackedDataset, sample_windows: int) -> dict:
    checked_queries = 0
    reset_queries = 0
    for index in range(min(sample_windows, len(dataset))):
        x, _ = dataset.fetch_batch([index])
        mask = model.build_attention_mask(x)[0, 0].cpu()
        segment = torch.cumsum(x[0] == int(model.sep_id), dim=0)
        expected = (
            torch.arange(x.size(1)).unsqueeze(1)
            >= torch.arange(x.size(1)).unsqueeze(0)
        ) & (segment.unsqueeze(1) == segment.unsqueeze(0))
        if not torch.equal(mask, expected):
            raise AssertionError(f"attention mask mismatch in packed window {index}")
        for query in range(x.size(1)):
            if int(x[0, query]) == PAD_ID:
                continue
            checked_queries += 1
            if query and int(x[0, query]) == int(model.sep_id):
                reset_queries += 1
                if bool(mask[query, query - 1]):
                    raise AssertionError("separator query can attend across reset boundary")
            elif query and not bool(mask[query, query - 1]):
                raise AssertionError("within-segment query cannot attend previous token")
    return {
        "sampled_windows": min(sample_windows, len(dataset)),
        "checked_nonpad_queries": checked_queries,
        "separator_reset_queries": reset_queries,
        "status": "passed",
    }


@torch.no_grad()
def diagnose(args: argparse.Namespace) -> dict:
    train_path = args.train.expanduser().resolve()
    test_path = args.test.expanduser().resolve()
    _, manifest_provenance = bind_dataset_manifest(
        args.manifest,
        expected_artifacts={
            "train_tokens": train_path,
            _evaluation_artifact_role(args.split): test_path,
        },
    )
    contract = resolve_vocabulary_contract(
        [train_path, test_path],
        configured_path=args.itos,
        configured_size=None,
    )
    reset_token_ids = frozenset(
        index for index, token in enumerate(contract.tokens) if token == "<SEP>"
    )
    if len(reset_token_ids) != 1:
        raise ValueError("diagnostic requires exactly one <SEP> token")

    state_dict, cfg, checkpoint_path = load_codon_checkpoint(
        args.run_dir, ckpt_name=args.checkpoint_name
    )
    checkpoint_dataset = bind_checkpoint_dataset(cfg, manifest_provenance)
    model = build_codon_model_from_cfg(cfg)
    model.load_state_dict(state_dict, strict=True)
    device = _device(args.device)
    model.to(device).eval()

    dataset = MmapPackedDataset(test_path)
    if dataset.is_dynamic:
        raise ValueError("packing-aware context diagnostics currently require fixed windows")
    chunked_windows = _packing_window_flags(args.packing_tsv, len(dataset))
    mask_audit = _mask_audit(model, dataset, args.mask_audit_windows)

    counts = fit_baselines(
        train_path,
        contract.size,
        args.alpha,
        reset_token_ids=reset_token_ids,
    )
    baseline_results, baseline_tokens, best_simple = evaluate_baselines(
        test_path,
        counts,
        contract.size,
        args.alpha,
        reset_token_ids=reset_token_ids,
    )
    _, bigram, trigram = counts

    windows = _parse_windows(args.context_windows)
    context_results = {}
    full_losses: list[np.ndarray] = []
    full_targets: list[np.ndarray] = []
    full_inputs: list[np.ndarray] = []
    full_rows: list[int] = []
    row_model = np.zeros(len(dataset), dtype=np.float64)
    row_trigram = np.zeros(len(dataset), dtype=np.float64)
    row_tokens = np.zeros(len(dataset), dtype=np.int64)

    for window in windows:
        label = "full" if window is None else str(window)
        print(f"[context] evaluating attention window {label}", flush=True)
        total_nll = 0.0
        total_tokens = 0
        for start in range(0, len(dataset), args.batch_size):
            indices = np.arange(start, min(start + args.batch_size, len(dataset)))
            xb, yb = dataset.fetch_batch(indices)
            logits, _ = model(
                xb.to(device),
                attention_window=window,
            )
            losses = F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)),
                yb.to(device).reshape(-1),
                ignore_index=PAD_ID,
                reduction="none",
            ).reshape(yb.shape)
            losses_np = losses.cpu().numpy()
            y_np = yb.numpy()
            valid = y_np != PAD_ID
            total_nll += float(losses_np[valid].sum())
            total_tokens += int(valid.sum())
            if window is None:
                x_np = xb.numpy()
                for row_offset, dataset_index in enumerate(indices):
                    valid_row = valid[row_offset]
                    model_row_losses = losses_np[row_offset]
                    trigram_losses = _trigram_nll(
                        x_np[row_offset],
                        y_np[row_offset],
                        trigram,
                        bigram,
                        alpha=args.alpha,
                        active_size=contract.size - 1,
                        reset_token_ids=reset_token_ids,
                    )
                    row_model[dataset_index] = float(model_row_losses[valid_row].sum())
                    row_trigram[dataset_index] = float(trigram_losses[valid_row].sum())
                    row_tokens[dataset_index] = int(valid_row.sum())
                    full_losses.append(model_row_losses)
                    full_targets.append(y_np[row_offset])
                    full_inputs.append(x_np[row_offset])
                    full_rows.append(int(dataset_index))
        mean_nll = total_nll / total_tokens
        context_results[label] = {
            "attention_window_input_tokens": window,
            "target_history_interpretation": (
                "full" if window is None else f"up_to_{window}_tokens_including_current_input"
            ),
            "nll": mean_nll,
            "perplexity": math.exp(mean_nll),
            "evaluated_tokens": total_tokens,
        }
        print(
            f"[context] window {label}: nll={mean_nll:.6f} "
            f"ppl={math.exp(mean_nll):.3f}",
            flush=True,
        )

    if "full" not in context_results:
        raise ValueError("context windows must include 'full' for decomposition")

    slices: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])
    sep_id = next(iter(reset_token_ids))
    for losses, targets, inputs, row_index in zip(
        full_losses, full_targets, full_inputs, full_rows, strict=True
    ):
        valid = targets != PAD_ID
        _add_slice(slices, "all", losses, valid)
        _add_slice(slices, "after_separator", losses, valid & (inputs == sep_id))
        _add_slice(
            slices,
            "window_with_chunk_continuation",
            losses,
            valid & np.full(len(valid), chunked_windows[row_index]),
        )
        _add_slice(
            slices,
            "window_without_chunk_continuation",
            losses,
            valid & np.full(len(valid), not chunked_windows[row_index]),
        )
        segment_position = 0
        position_masks: dict[str, np.ndarray] = {}
        for position in range(len(targets)):
            if not valid[position]:
                continue
            if int(inputs[position]) == sep_id:
                segment_position = 0
            name = _position_bin(segment_position)
            position_masks.setdefault(name, np.zeros(len(valid), dtype=bool))[position] = True
            segment_position += 1
        for name, mask in position_masks.items():
            _add_slice(slices, name, losses, mask)
        for token_id, token in enumerate(contract.tokens):
            mask = valid & (targets == token_id)
            if mask.any():
                _add_slice(slices, f"target_class_{_token_class(token)}", losses, mask)

    decomposition = {
        name: {
            "nll": values[0] / values[1],
            "perplexity": math.exp(values[0] / values[1]),
            "tokens": values[1],
        }
        for name, values in sorted(slices.items())
        if values[1]
    }
    paired = _bootstrap_paired_rows(
        row_model,
        row_trigram,
        row_tokens,
        seed=args.seed,
        samples=args.bootstrap_samples,
    )
    return {
        "schema_version": 1,
        "status": "diagnostic_complete",
        "evaluation_split": args.split,
        "checkpoint": artifact_provenance(checkpoint_path),
        "checkpoint_dataset": checkpoint_dataset,
        "dataset_manifest": manifest_provenance,
        "vocabulary": contract.provenance(),
        "packing": {
            "metadata": artifact_provenance(args.packing_tsv)
            if args.packing_tsv
            else None,
            "windows_with_chunk_continuation": int(chunked_windows.sum()),
            "total_windows": len(dataset),
        },
        "attention_mask_audit": mask_audit,
        "markov": {
            "history_reset_token_ids": sorted(reset_token_ids),
            "history_reset_tokens": [contract.tokens[index] for index in reset_token_ids],
            "evaluated_tokens": baseline_tokens,
            "best_simple_baseline": best_simple,
            "results": baseline_results,
        },
        "context_ablation": context_results,
        "loss_decomposition": decomposition,
        "paired_codonlm_vs_trigram": paired,
    }


def _markdown(report: dict) -> str:
    lines = [
        "# Context Learning Diagnostic",
        "",
        "## Context Ablation",
        "",
        "| Input attention window | NLL | PPL |",
        "| ---: | ---: | ---: |",
    ]
    for name, result in report["context_ablation"].items():
        lines.append(f"| {name} | {result['nll']:.6f} | {result['perplexity']:.3f} |")
    lines.extend(
        [
            "",
            "## Segment-Aware Markov Baselines",
            "",
            "| Model | NLL | PPL |",
            "| --- | ---: | ---: |",
        ]
    )
    for name, result in report["markov"]["results"].items():
        lines.append(
            f"| {name} | {result['cross_entropy_nats']:.6f} | {result['perplexity']:.3f} |"
        )
    paired = report["paired_codonlm_vs_trigram"]
    lines.extend(
        [
            "",
            "## Paired Gate",
            "",
            f"CodonLM minus trigram: `{paired['codonlm_minus_trigram_nats_per_token']:.6f}` "
            f"nats/token (95% packed-window bootstrap CI "
            f"`[{paired['ci95'][0]:.6f}, {paired['ci95'][1]:.6f}]`).",
            "",
            "## Loss Decomposition",
            "",
            "| Slice | Tokens | NLL | PPL |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for name, result in report["loss_decomposition"].items():
        lines.append(
            f"| {name} | {result['tokens']} | {result['nll']:.6f} | "
            f"{result['perplexity']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-name", default="best.pt")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("test", "validation"),
        default="test",
        help="Manifest role of the evaluation input (default: test).",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--itos", type=Path)
    parser.add_argument("--packing-tsv", type=Path)
    parser.add_argument("--context-windows", default="1,2,4,8,32,128,full")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--mask-audit-windows", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    report = diagnose(args)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    args.output_prefix.with_suffix(".md").write_text(_markdown(report))
    print(_markdown(report))


if __name__ == "__main__":
    main()
