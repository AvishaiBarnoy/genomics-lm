#!/usr/bin/env python3
"""Paired, reproducible raw-decoding audit of weight-space edits."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from scripts.eval_generation_prefix import _extract_frozen_split_cds
from src.codonlm.checkpoints import load_codon_model

RUNS = {
    "multi_offset_source": "corrected-multi-offset-heads-seed1337",
    "conservative": "corrected-weight-space-programming-seed1337",
    "primary_source": "corrected-codonlm-v1-batch64-lr-ablation-lr_1_5e4",
    "aggressive": "corrected-aggressive-weight-space-programming-seed1337",
    "targeted_terminal": "corrected-targeted-terminal-weight-edit-seed1337",
}
STOPS = {"TAA", "TAG", "TGA"}


def _summarize(rows: list[dict], name: str) -> dict:
    subset = [r for r in rows if r["model"] == name]
    n = len(subset)
    def mean(key):
        return float(np.mean([float(r[key]) for r in subset]))
    return {
        "n": n,
        "biological_stop_rate": mean("biological_stop"),
        "eos_rate": mean("eos"),
        "hard_cap_rate": mean("hard_cap"),
        "internal_stop_rate": mean("internal_stop"),
        "non_codon_rate": mean("non_codon"),
        "mean_generated_codons": mean("generated_codons"),
        "mean_gc": mean("gc"),
        "mean_gc3": mean("gc3"),
        "mean_adjacent_repeat_rate": mean("adjacent_repeat_rate"),
        "mean_unique_codon_fraction": mean("unique_codon_fraction"),
        "mean_length_ratio_to_reference": mean("length_ratio_to_reference"),
    }


def run(n_prompts: int, seeds: int, max_new: int, output: Path, temperature: float = 1.0,
        models: tuple[str, ...] = tuple(RUNS)):
    output.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    manifest = repo / "data/processed/corrected/corrected-codonlm-v1/genome/manifest.json"
    dna_file, source_meta = _extract_frozen_split_cds(output, manifest, "test", n_prompts, 1337)
    genes = [s.strip() for s in dna_file.read_text().splitlines()]
    itos = (repo / "runs" / RUNS["primary_source"] / "itos.txt").read_text().splitlines()
    stoi = {t: i for i, t in enumerate(itos)}
    assert all((repo / "runs" / r / "itos.txt").read_text().splitlines() == itos for r in RUNS.values())
    prefixes = []
    for g in genes:
        c = [g[i:i+3] for i in range(0, len(g) - 2, 3)]
        prefixes.append([stoi["<BOS_CDS>"]] + [stoi[t] for t in c[:3]])
    contexts = [p for p in prefixes for _ in range(seeds)]
    keys = [(i, j) for i in range(len(genes)) for j in range(seeds)]
    uniforms = np.empty((len(keys), max_new), dtype=np.float64)
    for row, (i, j) in enumerate(keys):
        digest = hashlib.sha256(f"wsp:{i}:{j}:1337".encode()).digest()
        uniforms[row] = np.random.default_rng(int.from_bytes(digest[:8], "big")).random(max_new)
    ctx = torch.tensor(contexts, dtype=torch.long)
    stop_ids = {stoi[t] for t in STOPS}
    eos_id = stoi["<EOS_CDS>"]
    rows = []
    for label in models:
        run_id = RUNS[label]
        model, cfg, ckpt_path = load_codon_model(repo / "runs" / run_id, device="cpu")
        ids = ctx.clone()
        ended = np.zeros(len(keys), dtype=bool)
        reasons = ["hard_cap"] * len(keys)
        generated = [[] for _ in keys]
        with torch.no_grad():
            for step in range(max_new):
                logits, _ = model(ids[:, -int(cfg["block_size"]):])
                probs = torch.softmax(logits[:, -1, :].double() / temperature, dim=-1).numpy()
                cdf = np.cumsum(probs, axis=1)
                picks = (cdf < uniforms[:, step, None]).sum(axis=1).clip(max=len(itos)-1)
                for i, pick in enumerate(picks):
                    if ended[i]:
                        continue
                    generated[i].append(int(pick))
                    if int(pick) in stop_ids:
                        reasons[i] = "biological_stop"
                        ended[i] = True
                    elif int(pick) == eos_id:
                        reasons[i] = "eos"
                        ended[i] = True
                ids = torch.cat((ids, torch.from_numpy(picks).long()[:, None]), dim=1)
                if (step + 1) % 25 == 0:
                    print(label, step + 1, "ended", int(ended.sum()), "/", len(keys), flush=True)
                if ended.all():
                    break
        for i, ((gene_idx, seed_idx), seq_ids) in enumerate(zip(keys, generated)):
            tokens = [itos[t] for t in seq_ids]
            codons = [t for t in tokens if len(t) == 3 and set(t) <= set("ACGT")]
            body = [c for c in codons if c not in STOPS]
            dna = "".join(body)
            total_bases = len(dna)
            gc_fraction = (dna.count("G") + dna.count("C")) / total_bases if total_bases else 0.0
            gc3 = sum(c[2] in "GC" for c in body) / len(body) if body else 0.0
            repeats = sum(a == b for a, b in zip(body, body[1:])) / max(1, len(body)-1)
            rows.append({
                "model": label, "run_id": run_id, "gene_idx": gene_idx, "seed_idx": seed_idx,
                "source_id": source_meta["selected_source_ids"][gene_idx],
                "reason": reasons[i], "biological_stop": int(reasons[i] == "biological_stop"),
                "eos": int(reasons[i] == "eos"), "hard_cap": int(reasons[i] == "hard_cap"),
                "internal_stop": int(any(t in STOPS for t in tokens[:-1])),
                "non_codon": int(any(len(t) != 3 or set(t) - set("ACGT") for t in tokens if t != "<EOS_CDS>")),
                "generated_codons": len(codons), "gc": gc_fraction, "gc3": gc3,
                "adjacent_repeat_rate": repeats,
                "unique_codon_fraction": len(set(body)) / len(body) if body else 0.0,
                "length_ratio_to_reference": len(body) / max(1, len(genes[gene_idx]) // 3 - 3),
                "tokens": " ".join(tokens), "generated_dna": dna,
            })
        print(label, _summarize(rows, label), flush=True)
        del model, ids
        gc.collect()
    with (output / "samples.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "protocol": {"n_prompts": len(genes), "seeds_per_prompt": seeds, "max_new": max_new,
                     "temperature": temperature, "topk": 0, "prefix_codons": 3,
                     "sampling": "common pre-drawn uniform per prompt/seed/step; CDF inverse",
                     "source": source_meta, "checkpoint_paths": {k: str(repo / "runs" / RUNS[k] / "checkpoints/best.pt") for k in models}},
        "models": {k: _summarize(rows, k) for k in models},
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-prompts", type=int, default=40)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--max-new", type=int, default=100)
    parser.add_argument("--models", nargs="+", choices=tuple(RUNS), default=list(RUNS))
    parser.add_argument("--output", type=Path, default=Path("runs/_summary/weight_space_paired_eval"))
    args = parser.parse_args()
    print(json.dumps(run(args.n_prompts, args.seeds, args.max_new, args.output, models=tuple(args.models)), indent=2))
