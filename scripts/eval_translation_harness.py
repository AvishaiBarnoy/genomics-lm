#!/usr/bin/env python3
"""Generate CDS sequences from a checkpoint and produce a translation report.

Usage:
  python -m scripts.eval_translation_harness \\
      --run_dir runs/2026-06-12_stage2.5_6L4H_d384_e5 \\
      --num_seqs 32 --max_codons 128 --seed 42

Outputs:
  runs/<RUN_ID>/scores/translation_report.csv   (one row per sequence)
  runs/<RUN_ID>/scores/translation_summary.json  (aggregate statistics)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from scripts.query_model import (
    _load_checkpoint,
    _load_vocab,
    build_model_from_state,
    generate,
)
from src.eval.translation import TranslationReport, batch_translate


def dev() -> torch.device:
    return (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )


def _generate_sequences(
    model,
    device: torch.device,
    itos: list[str],
    stoi: dict[str, int],
    *,
    num_seqs: int,
    max_codons: int,
    temperature: float,
    top_k: int,
    seed: int,
) -> list[list[str]]:
    """Generate *num_seqs* sequences, each up to *max_codons* new tokens."""
    torch.manual_seed(seed)
    bos_id = stoi.get("<BOS_CDS>", 1)
    eos_id = stoi.get("<EOS_CDS>")

    sequences: list[list[str]] = []
    for _ in range(num_seqs):
        ids = generate(
            model,
            device,
            ctx_ids=[bos_id],
            max_new=max_codons,
            temperature=temperature,
            topk=top_k,
            eos_idx=eos_id,
        )
        # Convert ids → codon strings
        tokens = [itos[i] if 0 <= i < len(itos) else f"<{i}>" for i in ids]
        sequences.append(tokens)
    return sequences


def _summarise(reports: list[TranslationReport]) -> dict:
    """Compute aggregate statistics from a batch of reports."""
    n = len(reports)
    if n == 0:
        return {}
    return {
        "num_sequences": n,
        "mean_translated_length": round(
            sum(r.translated_length for r in reports) / n, 2
        ),
        "frac_starts_with_met": round(
            sum(r.starts_with_met for r in reports) / n, 4
        ),
        "frac_has_terminal_stop": round(
            sum(r.has_terminal_stop for r in reports) / n, 4
        ),
        "frac_valid_orf": round(
            sum(r.is_valid_orf for r in reports) / n, 4
        ),
        "total_internal_stops": sum(r.internal_stop_count for r in reports),
        "total_invalid_codons": sum(r.num_invalid_codons for r in reports),
        "mean_codons_per_seq": round(
            sum(r.num_codons for r in reports) / n, 2
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate CDS from a checkpoint and evaluate translation quality."
    )
    ap.add_argument("--run_dir", required=True, help="Path to the run directory")
    ap.add_argument("--num_seqs", type=int, default=32, help="Number of sequences to generate")
    ap.add_argument("--max_codons", type=int, default=128, help="Max new codons per sequence")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out_dir",
        default=None,
        help="Output directory (default: <run_dir>/scores)",
    )
    args = ap.parse_args()

    if args.num_seqs <= 0 or args.max_codons <= 0:
        ap.error("--num_seqs and --max_codons must be positive")

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "scores"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load model ──────────────────────────────────────────────────────
    print(f"[translation-harness] loading checkpoint from {run_dir}")
    itos, stoi = _load_vocab(run_dir)
    state_dict, cfg = _load_checkpoint(run_dir)
    model = build_model_from_state(state_dict, cfg)
    device = dev()
    model.to(device)
    print(f"[translation-harness] model on {device}, vocab={len(itos)}")

    # ── generate ────────────────────────────────────────────────────────
    print(
        f"[translation-harness] generating {args.num_seqs} sequences "
        f"(max_codons={args.max_codons}, seed={args.seed}, T={args.temperature}, "
        f"top_k={args.top_k})"
    )
    sequences = _generate_sequences(
        model,
        device,
        itos,
        stoi,
        num_seqs=args.num_seqs,
        max_codons=args.max_codons,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
    )

    # ── translate & evaluate ────────────────────────────────────────────
    reports = batch_translate(sequences)

    # ── write CSV ───────────────────────────────────────────────────────
    csv_path = out_dir / "translation_report.csv"
    fieldnames = list(reports[0].to_dict().keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for report in reports:
            writer.writerow(report.to_dict())
    print(f"[translation-harness] wrote {csv_path} ({len(reports)} rows)")

    # ── write summary JSON ──────────────────────────────────────────────
    summary = _summarise(reports)
    summary_path = out_dir / "translation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[translation-harness] wrote {summary_path}")
    print(f"[translation-harness] summary: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
