#!/usr/bin/env python3
"""Structured-prefix generation experiment for the structured_generation track.

This is a small harness around the existing CodonLM + ProteinCritic stack. It
generates continuations from three known structured bacterial protein prefixes
and writes CSV/Markdown outputs comparable to the main generative design loop.

ESMFold submission is optional because it requires network access.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from scripts.generative_design_loop import (
    STOP_CODONS,
    dev,
    esm_fold,
    kmer_diversity,
    load_codon_lm,
    load_critic,
    pairwise_identity,
    score_with_critic,
    translate_dna,
)


PREFIXES = {
    # Approximate codon prompts for well-structured bacterial enzyme families.
    # They are prompts, not reference reconstruction targets.
    "dhfr_folA": {
        "family": "Dihydrofolate reductase / FolA-like",
        "dna": "ATGACCCTGTCTATTCTGGTTGCTGGTGCTGGTAAAGCTGGTGTT",
    },
    "tem1_betalactamase": {
        "family": "TEM-1 beta-lactamase-like",
        "dna": "ATGAGTATTCAACATTTCCGTGTCGCCCTGTTCGTTTTCGGT",
    },
    "tpiA": {
        "family": "Triosephosphate isomerase / TIM-barrel-like",
        "dna": "ATGAAAGTTATCGCTGGTGCTGGTACCGGTAAAGTTGTTGCT",
    },
}


def _prefix_ids(dna: str, stoi: dict[str, int]) -> list[int]:
    dna = dna.strip().upper().replace("U", "T")
    usable = len(dna) // 3 * 3
    if usable == 0:
        raise ValueError("prefix DNA must contain at least one complete codon")
    ids = [stoi.get("<BOS_CDS>", 1)]
    for i in range(0, usable, 3):
        codon = dna[i:i + 3]
        if codon not in stoi:
            raise ValueError(f"unknown prefix codon: {codon}")
        ids.append(stoi[codon])
    return ids


@torch.no_grad()
def generate_from_prefix(
    model,
    device: torch.device,
    stoi: dict[str, int],
    itos: list[str],
    prefix_dna: str,
    max_new_codons: int,
    temperature: float,
    top_k: int,
    top_p: float,
    anneal_temp: bool,
) -> tuple[list[str], bool]:
    ids = _prefix_ids(prefix_dna, stoi)
    eos_idx = stoi.get("<EOS_CDS>")
    stop_ids = {stoi[c] for c in STOP_CODONS if c in stoi}
    block_size = getattr(model, "block_size", None)
    anneal_start = 50
    temp_floor = temperature * 0.7
    terminated = False

    for step in range(max_new_codons):
        ctx = ids[-block_size:] if block_size is not None else ids
        x = torch.tensor(ctx, dtype=torch.long, device=device).unsqueeze(0)
        logits, _ = model(x)
        logits = logits[0, -1].float()

        temp = temperature
        if anneal_temp and step >= anneal_start:
            frac = min((step - anneal_start) / max(max_new_codons - anneal_start, 1), 1.0)
            temp = temperature - frac * (temperature - temp_floor)
        logits = logits / max(temp, 1e-6)

        if top_p > 0.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumprobs = torch.cumsum(probs, dim=-1)
            remove = cumprobs - probs > top_p
            sorted_logits[remove] = float("-inf")
            logits = torch.empty_like(logits).scatter_(0, sorted_idx, sorted_logits)
        elif top_k > 0:
            topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < topk_vals[-1]] = float("-inf")

        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1).item()
        ids.append(next_id)
        if next_id == eos_idx or next_id in stop_ids:
            terminated = True
            break

    tokens = [itos[i] for i in ids if 0 <= i < len(itos)]
    codons = [
        tok for tok in tokens
        if len(tok) == 3 and tok.isalpha() and tok not in STOP_CODONS
    ]
    return codons, terminated


def _write_report(
    path: Path,
    rows: list[dict],
    elapsed: float,
    args: argparse.Namespace,
    esm_rows: list[dict],
) -> None:
    aa_seqs = [r["aa_seq"] for r in rows if r["aa_seq"]]
    stability = [float(r.get("stability_prob", 0.0)) for r in rows]
    family_conf = [float(r.get("family_top1_conf", 0.0)) for r in rows]
    term = sum(1 for r in rows if r["terminated"])
    lines = [
        "# Structured Prefix Experiment Report",
        "",
        f"Generated {len(rows)} sequences from {len(PREFIXES)} structured prefixes in {elapsed:.1f}s.",
        "",
        "| Setting | Value |",
        "|---|---|",
        f"| run_dir | `{args.run_dir}` |",
        f"| sequences_per_prefix | {args.sequences_per_prefix} |",
        f"| max_new_codons | {args.max_new_codons} |",
        f"| temperature | {args.temperature} |",
        f"| top_k | {args.top_k} |",
        f"| top_p | {args.top_p} |",
        f"| anneal_temp | {args.anneal_temp} |",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| terminated | {term}/{len(rows)} ({term / max(len(rows), 1) * 100:.1f}%) |",
        f"| mean stability_prob | {np.mean(stability):.3f} |",
        f"| max stability_prob | {np.max(stability):.3f} |",
        f"| mean family_top1_conf | {np.mean(family_conf):.4f} |",
        f"| mean pairwise AA identity | {pairwise_identity(aa_seqs) * 100:.1f}% |",
        f"| 3-mer AA k-mer coverage | {kmer_diversity(aa_seqs) * 100:.2f}% |",
        "",
        "## By Prefix",
        "",
        "| Prefix | n | terminated | mean stability | mean AA length |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in PREFIXES:
        group = [r for r in rows if r["prefix_id"] == key]
        if not group:
            continue
        g_stab = [float(r.get("stability_prob", 0.0)) for r in group]
        g_len = [int(r.get("n_aa", 0)) for r in group]
        g_term = sum(1 for r in group if r["terminated"])
        lines.append(
            f"| {key} | {len(group)} | {g_term} | {np.mean(g_stab):.3f} | {np.mean(g_len):.1f} |"
        )

    if esm_rows:
        lines += [
            "",
            "## ESMFold Results",
            "",
            "| Prefix | Seq ID | pLDDT mean | pLDDT min |",
            "|---|---:|---:|---:|",
        ]
        for r in esm_rows:
            lines.append(
                f"| {r.get('prefix_id', '')} | {r.get('seq_id', '')} | "
                f"{r.get('plddt_mean', float('nan')):.3f} | {r.get('plddt_min', float('nan')):.3f} |"
            )

    lines += [
        "",
        "## Interpretation",
        "",
        "This experiment tests whether a fold-family prompt can bias CodonLM toward more structured continuations without retraining. It should be compared against the BOS-only and critic-filtered design-loop reports. If ESMFold remains low, the result supports the broader structured-generation finding: sampling controls alter sequence statistics and critic scores, but do not provide a true structural training signal.",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate from known structured protein prefixes.")
    ap.add_argument("--run_dir", default="runs/2026-06-15_stage2.6_10L8H_d384_e10")
    ap.add_argument("--critic_ckpt", default="runs/protein_critic/checkpoints/best_critic.pt")
    ap.add_argument("--critic_cfg", default="configs/protein_critic.yaml")
    ap.add_argument("--sequences_per_prefix", type=int, default=10)
    ap.add_argument("--max_new_codons", type=int, default=240)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--top_p", type=float, default=0.0)
    ap.add_argument("--anneal_temp", action="store_true")
    ap.add_argument("--esm_fold_top", type=int, default=0)
    ap.add_argument("--out_dir", default="outputs/reports/structured_prefix_experiment")
    args = ap.parse_args()

    device = dev()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    codon_model, itos, stoi = load_codon_lm(args.run_dir, device)
    critic_model, tokenizer, task_dims = load_critic(args.critic_ckpt, args.critic_cfg, device)

    rows: list[dict] = []
    start = time.time()
    seq_id = 0
    for prefix_id, meta in PREFIXES.items():
        for local_idx in range(args.sequences_per_prefix):
            codons, terminated = generate_from_prefix(
                codon_model,
                device,
                stoi,
                itos,
                meta["dna"],
                args.max_new_codons,
                args.temperature,
                args.top_k,
                args.top_p,
                args.anneal_temp,
            )
            dna_seq = "".join(codons)
            aa_seq, aa_terminated = translate_dna(dna_seq)
            row = {
                "seq_id": seq_id,
                "prefix_id": prefix_id,
                "prefix_family": meta["family"],
                "prefix_dna": meta["dna"],
                "local_idx": local_idx,
                "dna_seq": dna_seq,
                "aa_seq": aa_seq,
                "n_codons": len(codons),
                "n_aa": len(aa_seq),
                "terminated": terminated,
                "aa_terminated": aa_terminated,
            }
            if aa_seq:
                row.update(score_with_critic(critic_model, tokenizer, task_dims, aa_seq, device))
            rows.append(row)
            seq_id += 1
            print(f"[prefix] {prefix_id} {local_idx + 1}/{args.sequences_per_prefix}: "
                  f"len={row['n_aa']} terminated={terminated} "
                  f"stability={row.get('stability_prob', 0):.3f}")

    esm_rows: list[dict] = []
    if args.esm_fold_top > 0:
        top_rows = sorted(rows, key=lambda r: float(r.get("stability_prob", 0.0)), reverse=True)[:args.esm_fold_top]
        for rank, row in enumerate(top_rows, 1):
            fold = esm_fold(row["aa_seq"])
            if not fold:
                continue
            pdb_path = out_dir / f"top_{rank}_seq{row['seq_id']}.pdb"
            pdb_path.write_text(fold["pdb_text"])
            esm_rows.append({
                "rank": rank,
                "seq_id": row["seq_id"],
                "prefix_id": row["prefix_id"],
                "plddt_mean": fold["plddt_mean"],
                "plddt_min": fold["plddt_min"],
                "plddt_max": fold["plddt_max"],
                "pdb_path": str(pdb_path),
            })

    csv_path = out_dir / "structured_prefix_library.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, list) else v for k, v in row.items()})

    if esm_rows:
        esm_csv = out_dir / "esm_fold_results.csv"
        with esm_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(esm_rows[0].keys()), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(esm_rows)

    report_path = out_dir / "structured_prefix_report.md"
    _write_report(report_path, rows, time.time() - start, args, esm_rows)
    print(f"[prefix] wrote {csv_path}")
    print(f"[prefix] wrote {report_path}")


if __name__ == "__main__":
    main()
