#!/usr/bin/env python3
"""Evaluate generated proteins vs length-matched real proteins using frozen ESM-2 (8M).

Usage:
    python -m scripts.eval_esm2_critic \\
        --run_dir runs/corrected-codonlm-v1-batch64-lr15-seed2027 \\
        --num_seqs 32 --max_codons 128 --num_positions 16 --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from scripts._shared import resolve_run
from scripts.query_model import (
    _load_vocab,
    generate,
)
from src.codonlm.checkpoints import build_codon_model_from_cfg, load_codon_checkpoint
from src.codonlm.metrics_io import write_merge_metrics
from src.eval.protein_critic_esm import (
    ESM2ProteinCritic,
    extract_length_matched_validation_proteins,
    run_esm2_critic_benchmark,
)
from src.eval.translation import translate_codons


def get_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    # CPU is fastest for sequential auto-regressive decoding on Mac
    return torch.device("cpu")


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
    """Generate *num_seqs* sequences at batch size 1 with fixed seed."""
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
        tokens = [itos[i] if 0 <= i < len(itos) else f"<{i}>" for i in ids]
        sequences.append(tokens)
    return sequences


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score generated proteins against real validation proteins using ESM-2 (8M)."
    )
    parser.add_argument("--run_dir", help="outputs/checkpoints/<RUN_ID> or runs/<RUN_ID>")
    parser.add_argument("--run_id", help="Run id (alternative to --run_dir)")
    parser.add_argument(
        "--checkpoint-name",
        default="best.pt",
        help="Checkpoint filename under the run directory (default: best.pt)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/corrected/corrected-codonlm-v1/genome/manifest.json"),
        help="Path to dataset manifest.json containing validation records",
    )
    parser.add_argument(
        "--num_seqs",
        type=int,
        default=32,
        help="Number of generated proteins and real proteins to score (default: 32)",
    )
    parser.add_argument(
        "--max_codons",
        type=int,
        default=128,
        help="Max codons generated per sequence (default: 128)",
    )
    parser.add_argument(
        "--num_positions",
        type=int,
        default=16,
        help="Random residue positions to evaluate per protein (default: 16)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=0,
        help="Top-K sampling constraint (default: 0 = unconstrained sampling)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Fixed seed for generation and position subsampling (default: 42)",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "mps", "cuda"),
        default=None,
        help="PyTorch device",
    )
    parser.add_argument(
        "--out_json",
        type=Path,
        default=None,
        help="Optional path to write benchmark JSON results",
    )
    args = parser.parse_args()

    run_id, run_dir = resolve_run(args.run_id, args.run_dir)
    device = get_device(args.device)

    # 1. Load CodonLM generator
    print(f"[critic] Loading generator checkpoint from {run_dir} ...")
    itos, stoi = _load_vocab(run_dir)
    state_dict, cfg, checkpoint_path = load_codon_checkpoint(
        run_dir, ckpt_name=args.checkpoint_name
    )
    model = build_codon_model_from_cfg(cfg)
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()

    # 2. Generate 32 CDS sequences
    print(
        f"[critic] Generating {args.num_seqs} sequences of up to {args.max_codons} codons "
        f"(seed={args.seed}, T={args.temperature}, top_k={args.top_k}) ..."
    )
    raw_sequences = _generate_sequences(
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

    # 3. Translate generated sequences to proteins
    print("[critic] Translating generated coding sequences ...")
    generated_proteins: list[tuple[str, str]] = []
    target_lengths: list[int] = []
    for idx, tokens in enumerate(raw_sequences):
        report = translate_codons(tokens)
        prot = report.protein_sequence
        if not prot:
            # Fallback if unviable
            prot = "M"
        generated_proteins.append((f"gen_{idx:02d}", prot))
        target_lengths.append(len(prot))

    # 4. Extract length-matched real validation proteins
    print(f"[critic] Extracting {args.num_seqs} length-matched real validation proteins ...")
    real_proteins = extract_length_matched_validation_proteins(
        args.manifest,
        target_lengths,
        seed=args.seed,
    )

    # 5. Load ESM-2 Critic
    print(f"[critic] Loading frozen ESM-2 (8M) model on {device} ...")
    critic = ESM2ProteinCritic(device=device)

    # 6. Score all 64 proteins
    print(
        f"[critic] Scoring 32 generated and 32 real proteins (16 masked positions per protein) ..."
    )
    summary = run_esm2_critic_benchmark(
        critic,
        generated_proteins,
        real_proteins,
        num_positions=args.num_positions,
        seed=args.seed,
    )

    print("\n" + "=" * 65)
    print(" ESM-2 (8M) EXTERNAL PROTEIN CRITIC BENCHMARK")
    print("=" * 65)
    print(f"Run ID:                 {run_id}")
    print(f"Scored sequences:       {summary.num_generated} generated vs {summary.num_real} real (total: {len(summary.results)})")
    print(f"Evaluated positions:    {summary.num_positions_per_protein} per protein")
    print(f"Fixed seed:             {summary.seed}")
    print(f"All scores finite:      {summary.all_scores_finite}")
    print("-" * 65)
    print(f"Real Proteins Score:    {summary.mean_real_score:.4f} ± {summary.std_real_score:.4f} (median: {summary.median_real_score:.4f})")
    print(f"Generated Score:        {summary.mean_generated_score:.4f} ± {summary.std_generated_score:.4f} (median: {summary.median_generated_score:.4f})")
    print("-" * 65)
    print(f"Delta Median Score:     {summary.delta_median_score:+.4f} (Real - Generated)")
    print(f"Delta Mean Score:       {summary.delta_mean_score:+.4f}")
    if summary.p_value is not None:
        print(f"Mann-Whitney U:         U={summary.mann_whitney_u:.1f}, p={summary.p_value:.2e}")
    print("=" * 65)

    if summary.success:
        print(f"RESULT: PASS (All 64 scores finite AND Median(Real) > Median(Generated))")
        print("Conclusion: External ESM-2 critic confirms natural proteins have higher sequence plausibility than generated candidates.")
    else:
        print(f"RESULT: FAIL (All scores finite: {summary.all_scores_finite}, Delta Median: {summary.delta_median_score:+.4f})")
    print("=" * 65 + "\n")

    # Save output JSON
    out_dict = summary.to_dict()
    out_dict["run_id"] = run_id
    out_dict["checkpoint_name"] = args.checkpoint_name

    scores_dir = run_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    scores_file = scores_dir / "esm2_critic_benchmark.json"
    scores_file.write_text(json.dumps(out_dict, indent=2) + "\n")
    print(f"[critic] Saved report to {scores_file}")

    metrics_path = scores_dir / "metrics.json"
    write_merge_metrics(
        metrics_path,
        {
            "esm2_critic_real_median_score": summary.median_real_score,
            "esm2_critic_generated_median_score": summary.median_generated_score,
            "esm2_critic_delta_median": summary.delta_median_score,
            "esm2_critic_mann_whitney_p": summary.p_value,
        },
    )
    print(f"[critic] Merged summary metrics into {metrics_path}")

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(out_dict, indent=2) + "\n")
        print(f"[critic] Saved full report to {args.out_json}")

    return 0 if summary.success else 1


if __name__ == "__main__":
    sys.exit(main())
