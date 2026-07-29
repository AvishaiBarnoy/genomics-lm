#!/usr/bin/env python3
"""Compare stop-token probabilities on natural and generated CDS contexts."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from Bio import SeqIO

from scripts import query_model as Q
from scripts.eval_generation_prefix import (
    _cfg_from,
    _load_run_meta,
    _load_vocab_for_run,
    _model_spec_from,
    _resolve_cds_dna_path,
    _select_device,
)
from src.codonlm.generate import STOP_CODONS
from src.codonlm.codon_tokenize import tokenize_cds_fragments


def _token_probabilities(
    model,
    token_ids: list[int],
    positions: list[tuple[str, int]],
    stop_ids: list[int],
    eos_id: int | None,
    device: torch.device,
) -> list[dict]:
    if len(token_ids) < 1:
        return []
    max_length = int(getattr(model, "block_size", len(token_ids)))
    offset = max(0, len(token_ids) - max_length)
    context = token_ids[offset:]
    with torch.no_grad():
        logits, _ = model(torch.tensor([context], dtype=torch.long, device=device))
        probabilities = torch.softmax(logits[0].float(), dim=-1)
    rows = []
    for label, original_index in positions:
        local_index = original_index - offset
        if not 0 <= local_index < probabilities.shape[0]:
            continue
        probs = probabilities[local_index]
        order = torch.argsort(probs, descending=True)
        stop_probability = float(probs[stop_ids].sum().item())
        eos_probability = float(probs[eos_id].item()) if eos_id is not None else 0.0
        ranks = [
            int((order == token_id).nonzero(as_tuple=False)[0].item()) + 1
            for token_id in stop_ids + ([eos_id] if eos_id is not None else [])
        ]
        rows.append(
            {
                "position": label,
                "stop_probability": stop_probability,
                "eos_probability": eos_probability,
                "termination_probability": stop_probability + eos_probability,
                "best_termination_rank": min(ranks),
                "termination_in_top5": min(ranks) <= 5,
                "termination_in_top20": min(ranks) <= 20,
            }
        )
    return rows


def _summarize(rows: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["position"]].append(row)
    result = {}
    for position, selected in sorted(grouped.items()):
        probabilities = [row["termination_probability"] for row in selected]
        result[position] = {
            "n": len(selected),
            "mean_termination_probability": float(np.mean(probabilities)),
            "median_termination_probability": float(np.median(probabilities)),
            "mean_stop_probability": float(
                np.mean([row["stop_probability"] for row in selected])
            ),
            "mean_eos_probability": float(
                np.mean([row["eos_probability"] for row in selected])
            ),
            "top5_inclusion_rate": float(
                np.mean([row["termination_in_top5"] for row in selected])
            ),
            "top20_inclusion_rate": float(
                np.mean([row["termination_in_top20"] for row in selected])
            ),
            "median_best_termination_rank": float(
                np.median([row["best_termination_rank"] for row in selected])
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--generation-label", default="corrected_generation_primary")
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--max-genes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

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
    stop_ids = [stoi[token] for token in sorted(STOP_CODONS)]
    eos_id = stoi.get("<EOS_CDS>")

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

    natural_rows = []
    for sequence_index, dna in enumerate(dna_path.read_text().splitlines()):
        tokenized = tokenize_cds_fragments(dna, termination="none")
        if not tokenized.fragments:
            continue
        ids = max(tokenized.fragments, key=lambda fragment: fragment.codon_end).ids
        if len(ids) < 2:
            continue
        target_index = len(ids) - 1
        positions = [
            (f"distance_{distance}", target_index - distance)
            for distance in (1, 2, 4, 8, 16, 32)
            if target_index - distance >= 0
        ]
        for row in _token_probabilities(
            model, ids[:-1], positions, stop_ids, eos_id, device
        ):
            row["sequence_index"] = sequence_index
            natural_rows.append(row)

    generated_rows = []
    generated_fasta = (
        run_dir / "scores" / args.generation_label / "generated_protocols.fasta"
    )
    for record in SeqIO.parse(generated_fasta, "fasta"):
        ids = Q.dna_prefix_to_ids(str(record.seq), stoi)
        positions = [
            (f"length_{length}", length)
            for length in (32, 64, 128, 256)
            if length < len(ids)
        ]
        positions.append(("final", len(ids) - 1))
        for row in _token_probabilities(
            model, ids, positions, stop_ids, eos_id, device
        ):
            row["record_id"] = record.id
            row["protocol"] = (
                "raw_model"
                if record.id.startswith("raw_model_")
                else "cds_constrained"
            )
            generated_rows.append(row)

    generated_summary = {}
    for protocol in ("raw_model", "cds_constrained"):
        generated_summary[protocol] = _summarize(
            [row for row in generated_rows if row["protocol"] == protocol]
        )
    report = {
        "schema_version": 1,
        "run_id": args.run_id,
        "checkpoint": str(checkpoint_path.resolve()),
        "source_data": source_provenance,
        "tokens": {
            "stop": {token: stoi[token] for token in sorted(STOP_CODONS)},
            "eos": eos_id,
        },
        "natural_teacher_forced": _summarize(natural_rows),
        "generated": generated_summary,
        "rows": {"natural": natural_rows, "generated": generated_rows},
    }
    output = args.output or (
        run_dir / "scores" / args.generation_label / "termination_diagnostic.json"
    )
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"[termination-diagnostic] wrote {output}")


if __name__ == "__main__":
    main()
