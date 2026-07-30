#!/usr/bin/env python3
"""Run a small raw-decoding ablation for natural CodonLM termination."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from scripts import query_model as Q
from scripts.eval_generation_prefix import (
    _cfg_from,
    _load_run_meta,
    _load_vocab_for_run,
    _model_spec_from,
    _resolve_cds_dna_path,
    _select_device,
    _set_seed,
)
from src.codonlm.generate import generate_model_raw


VARIANTS = {
    "temperature_0.8_unrestricted": {"temperature": 0.8, "topk": 0},
    "temperature_1.0_unrestricted": {"temperature": 1.0, "topk": 0},
    "temperature_1.0_topk20": {"temperature": 1.0, "topk": 20},
}


def _sample_seed(base_seed: int, sequence_index: int, variant: str) -> int:
    payload = f"{base_seed}:{sequence_index}:{variant}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _summary(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "natural_stop_rate": float(
            np.mean([row["had_terminal_stop"] for row in rows])
        ),
        "eos_rate": float(np.mean([row["stop_reason"] == "eos" for row in rows])),
        "hard_cap_rate": float(np.mean([row["hit_hard_cap"] for row in rows])),
        "mean_generated_tokens": float(
            np.mean([row["generated_tokens"] for row in rows])
        ),
        "mean_generated_codons": float(
            np.mean([row["generated_codons"] for row in rows])
        ),
        "mean_gc_fraction": float(np.mean([row["gc_fraction"] for row in rows])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--max-genes", type=int, default=10)
    parser.add_argument("--max-new", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--variants",
        default=",".join(VARIANTS),
        help="Comma-separated decoding variants to run.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    selected_variants = [name.strip() for name in args.variants.split(",") if name.strip()]
    unknown_variants = sorted(set(selected_variants) - set(VARIANTS))
    if unknown_variants:
        parser.error(f"unknown variants: {', '.join(unknown_variants)}")
    if not selected_variants:
        parser.error("--variants must select at least one variant")

    repo = Path(__file__).resolve().parents[1]
    run_dir = repo / "runs" / args.run_id
    checkpoint_path = run_dir / "checkpoints" / args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    meta = _load_run_meta(run_dir)
    cfg = _cfg_from(meta, checkpoint)
    model = Q.build_model_from_state(
        state_dict, _model_spec_from(meta, checkpoint), checkpoint=checkpoint
    )
    device = _select_device(args.device)
    model.to(device).eval()
    itos, stoi = _load_vocab_for_run(run_dir, repo, cfg)
    dna_path, source_provenance = _resolve_cds_dna_path(
        repo,
        run_dir,
        cfg,
        max_genes=args.max_genes,
        seed=args.seed,
        dataset_manifest=None,
        source_split="test",
    )
    if dna_path is None:
        raise SystemExit("could not resolve frozen test CDS records")

    rows = []
    sequences = []
    for sequence_index, dna in enumerate(dna_path.read_text().splitlines()):
        prefix_ids = Q.dna_prefix_to_ids(dna[:3], stoi)
        for variant in selected_variants:
            parameters = VARIANTS[variant]
            sample_seed = _sample_seed(args.seed, sequence_index, variant)
            _set_seed(sample_seed)
            generated_ids, info = generate_model_raw(
                model,
                device,
                prefix_ids,
                stoi,
                itos,
                max_new_tokens=args.max_new,
                **parameters,
            )
            tokens = Q.ids_to_codons(generated_ids, itos)
            codons = [
                token
                for token in tokens
                if len(token) == 3 and set(token) <= set("ACGT")
            ]
            rows.append(
                {
                    "variant": variant,
                    "sequence_index": sequence_index,
                    "sample_seed": sample_seed,
                    **parameters,
                    **info,
                    "gc_fraction": (
                        sum(base in "GC" for base in "".join(codons))
                        / max(1, len("".join(codons)))
                    ),
                }
            )
            sequences.append(
                (f"{variant}_sequence{sequence_index}_seed{sample_seed}", "".join(codons))
            )

    report = {
        "schema_version": 1,
        "run_id": args.run_id,
        "checkpoint": str(checkpoint_path.resolve()),
        "source_data": source_provenance,
        "max_new_tokens": args.max_new,
        "summary": {
            variant: _summary([row for row in rows if row["variant"] == variant])
            for variant in selected_variants
        },
        "rows": rows,
    }
    output = args.output or (
        run_dir
        / "scores"
        / "corrected_generation_primary"
        / "decoding_termination_ablation.json"
    )
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    fasta_path = output.with_suffix(".fasta")
    fasta_path.write_text(
        "".join(f">{record_id}\n{sequence}\n" for record_id, sequence in sequences)
    )
    print(f"[decoding-ablation] wrote {output} and {fasta_path}")


if __name__ == "__main__":
    main()
