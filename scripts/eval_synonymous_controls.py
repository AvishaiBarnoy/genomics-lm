#!/usr/bin/env python3
"""CLI runner for synonymous control paired likelihood evaluation.

Usage:
    python -m scripts.eval_synonymous_controls \\
        --run_dir runs/corrected-codonlm-v1-batch64-lr15-seed2027 \\
        --num_genes 100 \\
        --mode uniform_alternative
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from scripts._shared import resolve_run
from src.codonlm.checkpoints import build_codon_model_from_cfg, load_codon_checkpoint
from src.codonlm.metrics_io import write_merge_metrics
from src.codonlm.training.vocabulary import load_itos
from src.eval.synonymous_controls import (
    extract_validation_genes,
    run_synonymous_pilot,
)


def get_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate biological codon choice via paired synonymous controls."
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
        "--num_genes",
        type=int,
        default=100,
        help="Number of validation genes to evaluate (default: 100)",
    )
    parser.add_argument(
        "--mode",
        choices=("uniform_alternative", "uniform_any", "within_gene_shuffle"),
        default="uniform_alternative",
        help="Synonymous shuffling mode (default: uniform_alternative)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="RNG seed for gene selection and synonymous mutation (default: 1337)",
    )
    parser.add_argument(
        "--min_codons",
        type=int,
        default=30,
        help="Minimum codons per gene (default: 30)",
    )
    parser.add_argument(
        "--max_codons",
        type=int,
        default=500,
        help="Maximum codons per gene to fit context window (default: 500)",
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
        help="Optional path to save full JSON results",
    )
    parser.add_argument(
        "--update_scores",
        action="store_true",
        default=True,
        help="Save summary to <run_dir>/scores/synonymous_controls_pilot.json (default: True)",
    )
    parser.add_argument(
        "--random_weights",
        action="store_true",
        default=False,
        help="Evaluate on an untrained / randomly initialized model as a negative control",
    )
    args = parser.parse_args()

    run_id, run_dir = resolve_run(args.run_id, args.run_dir)
    device = get_device(args.device)

    state_dict, cfg, checkpoint_path = load_codon_checkpoint(
        run_dir, ckpt_name=args.checkpoint_name
    )
    model = build_codon_model_from_cfg(cfg)
    if args.random_weights:
        print(f"[pilot] NEGATIVE CONTROL: Using randomly initialized model (untrained baseline) ...")
        checkpoint_path = Path("untrained_random_init")
    else:
        print(f"[pilot] Loading checkpoint {args.checkpoint_name} from {run_dir} ...")
        model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()

    # Resolve vocabulary
    manifest_path = args.manifest.expanduser().resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest_data = json.loads(manifest_path.read_text())
    vocab_rel = manifest_data.get("artifacts", {}).get("vocabulary", {}).get("path", "itos.txt")
    vocab_path = manifest_path.parent / vocab_rel
    itos = load_itos(vocab_path)
    stoi = {tok: idx for idx, tok in enumerate(itos)}

    print(f"[pilot] Extracting {args.num_genes} validation genes ({args.min_codons}-{args.max_codons} codons) ...")
    genes = extract_validation_genes(
        manifest_path,
        num_genes=args.num_genes,
        min_codons=args.min_codons,
        max_codons=args.max_codons,
        seed=args.seed,
    )
    print(f"[pilot] Successfully sampled {len(genes)} intact validation genes.")

    print(f"[pilot] Running paired evaluation (mode={args.mode}, batch_size=1) on {device} ...")
    summary = run_synonymous_pilot(
        model,
        device,
        genes,
        itos=itos,
        stoi=stoi,
        mode=args.mode,
        seed=args.seed,
    )

    print("\n" + "=" * 65)
    print(" SYNONYMOUS CONTROLS EVALUATION REPORT")
    print("=" * 65)
    print(f"Run ID:            {run_id}")
    print(f"Evaluated genes:   {summary.num_genes}")
    print(f"Control mode:      {summary.mode}")
    print(f"Random seed:       {summary.seed}")
    print("-" * 65)
    print(f"Natural NLL:       {summary.mean_natural_nll:.4f} (median: {summary.median_natural_nll:.4f})")
    print(f"Control NLL:       {summary.mean_control_nll:.4f} (median: {summary.median_control_nll:.4f})")
    print(f"Natural PPL:       {summary.mean_natural_ppl:.2f}")
    print(f"Control PPL:       {summary.mean_control_ppl:.2f}")
    print("-" * 65)
    print(f"Median ΔNLL:       {summary.median_delta_nll:+.4f} nats/codon")
    print(f"Mean ΔNLL:         {summary.mean_delta_nll:+.4f} ± {summary.std_delta_nll:.4f}")
    print(f"IQR [Q25, Q75]:    [{summary.q25_delta_nll:+.4f}, {summary.q75_delta_nll:+.4f}]")
    print(f"Min / Max ΔNLL:    [{summary.min_delta_nll:+.4f}, {summary.max_delta_nll:+.4f}]")
    print(f"Win Rate:          {summary.win_rate * 100:.1f}% ({int(round(summary.win_rate * summary.num_genes))}/{summary.num_genes} genes natural > control)")
    if summary.wilcoxon_p_value is not None:
        print(f"Wilcoxon signed:   W={summary.wilcoxon_statistic:.1f}, p={summary.wilcoxon_p_value:.2e}")
    print("=" * 65)

    if summary.success:
        print(f"RESULT: PASS (Median ΔNLL = {summary.median_delta_nll:+.4f} > 0)")
        print("Conclusion: Natural bacterial codon choice has higher likelihood than synonymous variants.")
    else:
        print(f"RESULT: FAIL (Median ΔNLL = {summary.median_delta_nll:+.4f} <= 0)")
        print("Conclusion: Model does not favor natural codon choice over synonymous variants.")
    print("=" * 65 + "\n")

    # Save output JSON
    out_dict = summary.to_dict()
    out_dict["run_id"] = run_id
    out_dict["checkpoint_path"] = str(checkpoint_path)
    out_dict["manifest_path"] = str(manifest_path)

    if args.update_scores:
        scores_dir = run_dir / "scores"
        scores_dir.mkdir(parents=True, exist_ok=True)
        name_suffix = f"{args.mode}_random_init" if args.random_weights else args.mode
        scores_file = scores_dir / f"synonymous_controls_{name_suffix}.json"
        scores_file.write_text(json.dumps(out_dict, indent=2) + "\n")
        print(f"[pilot] Saved summary to {scores_file}")

        if not args.random_weights:
            metrics_path = scores_dir / "metrics.json"
            write_merge_metrics(
                metrics_path,
                {
                    f"synonymous_{args.mode}_median_delta_nll": summary.median_delta_nll,
                    f"synonymous_{args.mode}_mean_delta_nll": summary.mean_delta_nll,
                    f"synonymous_{args.mode}_win_rate": summary.win_rate,
                    f"synonymous_{args.mode}_wilcoxon_p": summary.wilcoxon_p_value,
                },
            )
            print(f"[pilot] Merged summary metrics into {metrics_path}")

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(out_dict, indent=2) + "\n")
        print(f"[pilot] Saved full report to {args.out_json}")

    return 0 if summary.success else 1


if __name__ == "__main__":
    sys.exit(main())
