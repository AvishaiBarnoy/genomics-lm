#!/usr/bin/env python3
"""Evaluate a CodonLM termination-distance head on a frozen data split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from scripts._shared import resolve_run
from scripts.evaluate_test import _find_test_npz, _find_validation_npz, dev
from src.codonlm.checkpoints import build_codon_model_from_cfg, load_codon_checkpoint
from src.codonlm.data_loading import (
    PackedDataset,
    dynamic_lm_collate_fn,
)
from src.codonlm.training.objectives import termination_distance_bucket_labels


def summarize_confusion(
    confusion: torch.Tensor,
    true_probability_sums: torch.Tensor,
) -> dict:
    counts = confusion.sum(dim=1)
    predicted = confusion.sum(dim=0)
    diagonal = confusion.diag().double()
    recall = diagonal / counts.clamp_min(1)
    precision = diagonal / predicted.clamp_min(1)
    total = confusion.sum().clamp_min(1)
    return {
        "evaluated_positions": int(confusion.sum()),
        "accuracy": float(diagonal.sum() / total),
        "balanced_accuracy": float(recall.mean()),
        "confusion_matrix": confusion.tolist(),
        "classes": [
            {
                "class": class_index,
                "count": int(counts[class_index]),
                "fraction": float(counts[class_index] / total),
                "recall": float(recall[class_index]),
                "precision": float(precision[class_index]),
                "mean_true_probability": float(
                    true_probability_sums[class_index]
                    / counts[class_index].clamp_min(1)
                ),
            }
            for class_index in range(confusion.shape[0])
        ],
    }


@torch.no_grad()
def evaluate(model, loader, device, cfg: dict) -> dict:
    bucket_edges = tuple(int(value) for value in cfg["termination_bucket_edges"])
    stop_ids = tuple(int(value) for value in cfg["termination_stop_ids"])
    n_classes = int(cfg["termination_n_classes"])
    class_weights = cfg.get("termination_class_weights")
    weights = (
        torch.tensor(class_weights, dtype=torch.float32, device=device)
        if class_weights is not None
        else None
    )
    confusion = torch.zeros((n_classes, n_classes), dtype=torch.long)
    true_probability_sums = torch.zeros(n_classes, dtype=torch.float64)
    loss_sum = 0.0
    loss_denominator = 0.0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        _, _, aux = model(xb, return_aux=True)
        logits = aux["termination_logits"].float()
        labels = termination_distance_bucket_labels(yb, stop_ids, bucket_edges)
        valid = labels != -100
        targets = labels[valid]
        selected = logits[valid]
        predictions = selected.argmax(dim=-1)
        batch_confusion = torch.bincount(
            targets * n_classes + predictions,
            minlength=n_classes * n_classes,
        ).reshape(n_classes, n_classes)
        confusion += batch_confusion.cpu()
        loss_sum += float(
            F.cross_entropy(
                selected,
                targets,
                weight=weights,
                reduction="sum",
            ).item()
        )
        loss_denominator += float(
            weights[targets].sum().item() if weights is not None else targets.numel()
        )
        true_probabilities = (
            torch.softmax(selected, dim=-1)
            .gather(1, targets[:, None])
            .squeeze(1)
            .cpu()
            .double()
        )
        targets_cpu = targets.cpu()
        for class_index in range(n_classes):
            true_probability_sums[class_index] += true_probabilities[
                targets_cpu == class_index
            ].sum()

    return {
        "bucket_edges": list(bucket_edges),
        "stop_ids": list(stop_ids),
        "class_weights": class_weights,
        "weighted_cross_entropy": loss_sum / max(loss_denominator, 1.0),
        **summarize_confusion(confusion, true_probability_sums),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    run_id, run_dir = resolve_run(args.run_id, None)
    state_dict, cfg, checkpoint_path = load_codon_checkpoint(
        run_dir, ckpt_name=args.checkpoint
    )
    if not cfg.get("termination_loss_enabled"):
        raise ValueError("checkpoint configuration has no termination head")
    model = build_codon_model_from_cfg(cfg)
    model.load_state_dict(state_dict, strict=True)
    device = dev()
    model.to(device).eval()

    repo = Path(__file__).resolve().parents[1]
    if args.split == "validation":
        data_path = _find_validation_npz(cfg, repo)
    else:
        data_path = _find_test_npz(run_id, cfg, repo, None)
    dataset = PackedDataset(data_path)
    collate_fn = dynamic_lm_collate_fn if dataset.is_dynamic else None
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    report = {
        "schema_version": 1,
        "run_id": run_id,
        "checkpoint": str(checkpoint_path.resolve()),
        "split": args.split,
        "data": str(data_path.resolve()),
        "device": str(device),
        **evaluate(model, loader, device, cfg),
    }
    output = args.output or (
        run_dir / "scores" / "termination_head_evaluation" / f"{args.split}_metrics.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"[termination-head] wrote {output}")
    print(
        f"[termination-head] loss={report['weighted_cross_entropy']:.4f} "
        f"accuracy={report['accuracy']:.4f} "
        f"balanced_accuracy={report['balanced_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
