#!/usr/bin/env python3
"""
Benchmark ReD (Reset-and-Discard) vs Standard (Solve-to-completion) sampling.

This script compares two compute allocation policies for finding terminal stop codons:
1. Standard: Try prefix 1 until success or max_attempts, then prefix 2, etc.
2. ReD: Try all prefixes round-robin (τ=1), discarding solved ones, until budget is hit.

The metric is coverage@cost: number of unique prefixes solved vs total tokens spent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from scripts import query_model as Q
from src.codonlm.generate import batch_red_sampler, generate_cds_constrained


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--max_genes", type=int, default=50)
    ap.add_argument("--k", type=int, default=1, help="Prefix length in codons")
    ap.add_argument(
        "--global_budget", type=int, default=50000, help="Total tokens (codons) allowed"
    )
    ap.add_argument(
        "--hard_cap", type=int, default=300, help="Max tokens per single attempt"
    )
    ap.add_argument("--target_codons", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--ckpt", default="best.pt")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    run_dir = repo / "runs" / args.run_id
    out_dir = repo / "outputs" / "scores" / args.run_id / "red_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    itos, stoi = Q._load_vocab(run_dir)
    meta = json.loads((run_dir / "meta.json").read_text())
    weights_path = run_dir / args.ckpt
    if not weights_path.exists():
        weights_path = repo / "outputs" / "checkpoints" / args.run_id / args.ckpt

    ckpt = torch.load(weights_path, map_location=Q.dev())
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model = Q.build_model_from_state(state_dict, meta["model_spec"])
    device = Q.dev()
    model.to(device).eval()

    # Load prefixes
    dna_path = None
    manifest = run_dir / "combined_manifest.json"
    if manifest.exists():
        data = json.loads(manifest.read_text())
        if data.get("datasets"):
            dna_path = repo / data["datasets"][0]["dna"]
    if not dna_path or not dna_path.exists():
        raise SystemExit("Could not find DNA source file.")

    prefixes = []
    with open(dna_path) as f:
        for line in f:
            dna = line.strip().upper().replace("U", "T")
            if len(dna) >= 3 * args.k:
                prefix = dna[: 3 * args.k]
                prefixes.append(Q.dna_to_ids(prefix, stoi))
            if len(prefixes) >= args.max_genes:
                break

    print(f"Loaded {len(prefixes)} prefixes. Budget: {args.global_budget} tokens.")

    # --- Standard Policy (Solve-to-completion) ---
    print("\nRunning Standard Policy...")
    std_solved = {}
    std_tokens_log = [0]
    std_coverage_log = [0]
    total_tokens_std = 0

    for i, ctx in enumerate(prefixes):
        if total_tokens_std >= args.global_budget:
            break

        # Try until success or budget hit
        while total_tokens_std < args.global_budget:
            gen_ids, info = generate_cds_constrained(
                model,
                device,
                ctx,
                stoi,
                itos,
                args.target_codons,
                args.hard_cap,
                require_terminal_stop=True,
                temperature=args.temperature,
                topk=args.topk,
            )
            spent = info["generated_codons"]
            total_tokens_std += spent
            std_tokens_log.append(total_tokens_std)

            if info["had_terminal_stop"]:
                std_solved[i] = (gen_ids, info)
                std_coverage_log.append(len(std_solved))
                break
            else:
                std_coverage_log.append(len(std_solved))
                # Continue trying the same prefix (Standard Policy)

    # --- ReD Policy ---
    print("\nRunning ReD Policy...")
    red_solved, remaining, total_tokens_red = batch_red_sampler(
        model,
        device,
        prefixes,
        stoi,
        itos,
        args.target_codons,
        args.hard_cap,
        global_token_budget=args.global_budget,
        temperature=args.temperature,
        topk=args.topk,
    )

    # To get a curve for ReD, we need to log progress during batch_red_sampler.
    # Since I implemented batch_red_sampler as a whole, I'll re-run it step-by-step for the log.
    red_tokens_log = [0]
    red_coverage_log = [0]
    curr_red_tokens = 0
    curr_red_solved = set()
    active_tasks = [(list(ctx), i) for i, ctx in enumerate(prefixes)]

    while active_tasks and curr_red_tokens < args.global_budget:
        next_active = []
        for ctx, idx in active_tasks:
            if curr_red_tokens >= args.global_budget:
                break

            gen_ids, info = generate_cds_constrained(
                model,
                device,
                ctx,
                stoi,
                itos,
                args.target_codons,
                args.hard_cap,
                require_terminal_stop=True,
                temperature=args.temperature,
                topk=args.topk,
            )
            spent = info["generated_codons"]
            curr_red_tokens += spent
            red_tokens_log.append(curr_red_tokens)

            if info["had_terminal_stop"]:
                curr_red_solved.add(idx)
            red_coverage_log.append(len(curr_red_solved))

            if not info["had_terminal_stop"]:
                next_active.append((ctx, idx))
        active_tasks = next_active

    # --- Plotting & Results ---
    plt.figure(figsize=(10, 6))
    plt.step(
        std_tokens_log,
        std_coverage_log,
        label="Standard (Solve-to-completion)",
        where="post",
    )
    plt.step(
        red_tokens_log, red_coverage_log, label="ReD (Reset-and-Discard)", where="post"
    )
    plt.xlabel("Total Tokens (Inference Cost)")
    plt.ylabel("Coverage (Prefixes Terminated)")
    plt.title(f"ReD vs Standard Coverage@Cost (Run: {args.run_id}, k={args.k})")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plot_path = out_dir / "coverage_at_cost.png"
    plt.savefig(plot_path)
    print(f"\nSaved plot to {plot_path}")

    results = {
        "run_id": args.run_id,
        "global_budget": args.global_budget,
        "standard": {
            "solved": len(std_solved),
            "total_tokens": total_tokens_std,
            "coverage_rate": len(std_solved) / len(prefixes),
        },
        "red": {
            "solved": len(red_solved),
            "total_tokens": total_tokens_red,
            "coverage_rate": len(red_solved) / len(prefixes),
        },
    }

    json_path = out_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {json_path}")

    print("\nSummary:")
    print(
        f"Standard Solved: {len(std_solved)} / {len(prefixes)} ({results['standard']['coverage_rate']:.1%})"
    )
    print(
        f"ReD Solved:      {len(red_solved)} / {len(prefixes)} ({results['red']['coverage_rate']:.1%})"
    )


if __name__ == "__main__":
    main()
