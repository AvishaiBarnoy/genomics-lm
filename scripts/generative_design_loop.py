#!/usr/bin/env python3
"""
scripts/generative_design_loop.py — Generative Design Loop

Full pipeline:
  1. CodonLM (Stage 2.6) generates N candidate sequences using ReD sampling
     (Reset-and-Discard: restart until we get a properly terminated sequence)
  2. MultiTaskProteinCritic scores each sequence on:
       - Pfam family (1000 classes) → top-1/top-5 + confidence
       - EC function (500 classes)  → top-1/top-5 + confidence
       - Stability (binary)         → prob_stable
  3. ProteinLM log-likelihood scores (perplexity under sequence model)
  4. Diversity metrics across the generated library
  5. Optional ESMFold API → pLDDT structure confidence for top-K sequences
  6. Output: CSV + Markdown summary report

Usage:
    python -m scripts.generative_design_loop \\
        --run_dir runs/2026-06-15_stage2.6_10L8H_d384_e10 \\
        --critic_ckpt runs/protein_critic/checkpoints/best_critic.pt \\
        --critic_cfg configs/protein_critic.yaml \\
        --n_sequences 50 \\
        --max_attempts 20 \\
        --max_codons 300 \\
        --out_dir outputs/reports/generative_design \\
        [--esm_fold_top 5]   # submit top-5 to ESMFold API (requires internet)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml

# ────────────────────────────────────────────────────────────────────────────
# Codon translation table
# ────────────────────────────────────────────────────────────────────────────
CODON_TABLE = {
    "ATA": "I", "ATC": "I", "ATT": "I", "ATG": "M",
    "ACA": "T", "ACC": "T", "ACG": "T", "ACT": "T",
    "AAC": "N", "AAT": "N", "AAA": "K", "AAG": "K",
    "AGC": "S", "AGT": "S", "AGA": "R", "AGG": "R",
    "CTA": "L", "CTC": "L", "CTG": "L", "CTT": "L",
    "CCA": "P", "CCC": "P", "CCG": "P", "CCT": "P",
    "CAC": "H", "CAT": "H", "CAA": "Q", "CAG": "Q",
    "CGA": "R", "CGC": "R", "CGG": "R", "CGT": "R",
    "GTA": "V", "GTC": "V", "GTG": "V", "GTT": "V",
    "GCA": "A", "GCC": "A", "GCG": "A", "GCT": "A",
    "GAC": "D", "GAT": "D", "GAA": "E", "GAG": "E",
    "GGA": "G", "GGC": "G", "GGG": "G", "GGT": "G",
    "TCA": "S", "TCC": "S", "TCG": "S", "TCT": "S",
    "TTC": "F", "TTT": "F", "TTA": "L", "TTG": "L",
    "TAC": "Y", "TAT": "Y", "TAA": "_", "TAG": "_",
    "TGC": "C", "TGT": "C", "TGA": "_", "TGG": "W",
}
STOP_CODONS = {"TAA", "TAG", "TGA"}


def translate_dna(dna: str) -> tuple[str, bool]:
    """Translate DNA string to amino acid string.
    Returns (aa_seq, terminated) where terminated=True if a stop codon was hit."""
    aa, terminated = "", False
    for i in range(0, len(dna) - 2, 3):
        codon = dna[i:i+3].upper()
        aa_char = CODON_TABLE.get(codon, "X")
        if aa_char == "_":
            terminated = True
            break
        aa += aa_char
    return aa, terminated


# ────────────────────────────────────────────────────────────────────────────
# Device selection
# ────────────────────────────────────────────────────────────────────────────
def dev() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ────────────────────────────────────────────────────────────────────────────
# Load CodonLM
# ────────────────────────────────────────────────────────────────────────────
def load_codon_lm(run_dir: str, device: torch.device):
    from scripts.query_model import (
        _load_checkpoint as load_ckpt,
        _load_vocab as load_vocab,
        build_model_from_state,
    )
    run_path = Path(run_dir)
    itos, stoi = load_vocab(run_path)
    state_dict, cfg = load_ckpt(run_path)
    model = build_model_from_state(state_dict, cfg).to(device).eval()
    return model, itos, stoi


# ────────────────────────────────────────────────────────────────────────────
# ReD Sampling (Reset-and-Discard)
# ────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def red_generate(
    model,
    device: torch.device,
    stoi: dict,
    itos: list[str],
    max_codons: int = 300,
    max_attempts: int = 20,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.0,
    min_aa_length: int = 50,
    anneal_temp: bool = False,
) -> tuple[list[str], bool, int]:
    """Generate one properly-terminated sequence using Reset-and-Discard.

    Sequences shorter than min_aa_length are discarded and retried.

    Args:
        top_p:       Nucleus sampling probability mass (0 = disabled, use top_k).
        anneal_temp: If True, linearly anneal temperature from `temperature` → 0.7×
                     after the first 50 codons (structured cores are more conserved).

    Returns: (codon_list, terminated, n_attempts)
    """
    bos = stoi.get("<BOS_CDS>", 1)
    eos_idx = stoi.get("<EOS_CDS>")
    bio_stop_ids = {stoi[c] for c in STOP_CODONS if c in stoi}
    anneal_start = 50
    temp_floor = temperature * 0.7

    for attempt in range(1, max_attempts + 1):
        ids = [bos]
        terminated = False
        for step in range(max_codons):
            x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
            logits, _ = model(x)
            logits = logits[0, -1].float()

            # T1c: temperature annealing after first 50 codons
            t = temperature
            if anneal_temp and step >= anneal_start:
                frac = min((step - anneal_start) / max(max_codons - anneal_start, 1), 1.0)
                t = temperature - frac * (temperature - temp_floor)
            logits = logits / max(t, 1e-6)

            # T2b: nucleus (top-p) sampling — overrides top-k when enabled
            if top_p > 0.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumprobs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                # remove tokens whose cumulative prob exceeds top_p
                remove = cumprobs - torch.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove] = float("-inf")
                logits = torch.empty_like(logits).scatter_(0, sorted_idx, sorted_logits)
            elif top_k > 0:
                topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < topk_vals[-1]] = float("-inf")

            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            ids.append(next_id)

            # Stop conditions
            if eos_idx is not None and next_id == eos_idx:
                terminated = True
                break
            if next_id in bio_stop_ids:
                terminated = True
                break

        codon_tokens = [itos[i] for i in ids if 0 <= i < len(itos)]
        coding = [
            t for t in codon_tokens
            if len(t) == 3 and t.isalpha() and t not in STOP_CODONS
            and t not in {"<BOS_CDS>", "<EOS_CDS>", "<PAD>"}
        ]

        # Accept if terminated AND long enough — otherwise discard (ReD)
        if terminated and len(coding) >= min_aa_length:
            return coding, True, attempt

    # Max attempts exhausted — return whatever we have (not terminated / too short)
    return coding, False, max_attempts


# ────────────────────────────────────────────────────────────────────────────
# Load MultiTask ProteinCritic
# ────────────────────────────────────────────────────────────────────────────
def load_critic(ckpt_path: str, cfg_path: str, device: torch.device):
    from src.protein_lm.tokenizer import ProteinTokenizer
    from src.protein_lm.models_multi import MultiTaskProteinClassifier
    from src.protein_lm.config import ProteinClassifierConfig

    tokenizer = ProteinTokenizer()
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    state = torch.load(ckpt_path, map_location=device)
    state_dict = state.get("model_state_dict", state)
    state_dict = state_dict.get("model", state_dict)

    task_dims = {}
    if "heads.family.weight" in state_dict:
        task_dims["family"] = state_dict["heads.family.weight"].shape[0]
    if "heads.function.weight" in state_dict:
        task_dims["function"] = state_dict["heads.function.weight"].shape[0]
    if "heads.stability.weight" in state_dict:
        task_dims["stability"] = state_dict["heads.stability.weight"].shape[0]

    model_cfg = ProteinClassifierConfig(
        vocab_size=len(tokenizer.vocab),
        block_size=cfg.get("block_size", 512),
        n_layer=cfg.get("n_layer", 4),
        n_head=cfg.get("n_head", 4),
        n_embd=cfg.get("n_embd", 128),
        dropout=0.0,
        num_classes=0,
    )
    model = MultiTaskProteinClassifier(model_cfg, task_dims).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, tokenizer, task_dims


@torch.no_grad()
def score_with_critic(
    critic_model,
    tokenizer,
    task_dims: dict,
    aa_seq: str,
    device: torch.device,
) -> dict:
    """Run AA sequence through the MultiTask critic. Returns scores dict."""
    # Tokenize using the tokenizer's encode_sequence method
    ids = [tokenizer.bos_token_id]
    ids += tokenizer.encode_sequence(aa_seq)
    ids.append(tokenizer.eos_token_id)

    max_len = 512
    ids = ids[:max_len]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    logits_dict = critic_model(input_ids)
    scores = {}

    if "stability" in task_dims:
        stab_logits = logits_dict["stability"][0]
        stab_probs = torch.softmax(stab_logits, dim=-1)
        scores["stability_prob"] = stab_probs[-1].item()  # prob of stable class
        scores["stability_pred"] = stab_logits.argmax().item()

    if "family" in task_dims:
        fam_logits = logits_dict["family"][0]
        fam_probs = torch.softmax(fam_logits, dim=-1)
        top5_vals, top5_idx = torch.topk(fam_probs, min(5, task_dims["family"]))
        scores["family_top1"] = top5_idx[0].item()
        scores["family_top1_conf"] = top5_vals[0].item()
        scores["family_top5"] = top5_idx.tolist()
        scores["family_top5_conf"] = top5_vals.tolist()
        scores["family_entropy"] = -(fam_probs * (fam_probs + 1e-10).log()).sum().item()

    if "function" in task_dims:
        fn_logits = logits_dict["function"][0]
        fn_probs = torch.softmax(fn_logits, dim=-1)
        top5_vals, top5_idx = torch.topk(fn_probs, min(5, task_dims["function"]))
        scores["function_top1"] = top5_idx[0].item()
        scores["function_top1_conf"] = top5_vals[0].item()
        scores["function_top5"] = top5_idx.tolist()
        scores["function_top5_conf"] = top5_vals.tolist()
        scores["function_entropy"] = -(fn_probs * (fn_probs + 1e-10).log()).sum().item()

    return scores


# ────────────────────────────────────────────────────────────────────────────
# Diversity metrics
# ────────────────────────────────────────────────────────────────────────────
def pairwise_identity(seqs: list[str], max_pairs: int = 500) -> float:
    """Average pairwise sequence identity (fraction of identical positions)."""
    if len(seqs) < 2:
        return 1.0
    import random
    pairs = [(seqs[i], seqs[j]) for i in range(len(seqs)) for j in range(i+1, len(seqs))]
    if len(pairs) > max_pairs:
        pairs = random.sample(pairs, max_pairs)
    identities = []
    for a, b in pairs:
        min_len = min(len(a), len(b))
        if min_len == 0:
            continue
        matches = sum(x == y for x, y in zip(a[:min_len], b[:min_len]))
        identities.append(matches / min_len)
    return float(np.mean(identities)) if identities else 0.0


def kmer_diversity(seqs: list[str], k: int = 3) -> float:
    """Fraction of possible k-mers observed across all sequences (normalised)."""
    observed = set()
    for seq in seqs:
        for i in range(len(seq) - k + 1):
            observed.add(seq[i:i+k])
    total_possible = 20 ** k  # amino acid k-mers
    return len(observed) / total_possible


def gc_content(codon_seqs: list[list[str]]) -> list[float]:
    """GC content per sequence."""
    results = []
    for codons in codon_seqs:
        dna = "".join(codons)
        if not dna:
            results.append(0.0)
            continue
        gc = sum(1 for c in dna.upper() if c in "GC")
        results.append(gc / len(dna))
    return results


# ────────────────────────────────────────────────────────────────────────────
# Optional ESMFold API
# ────────────────────────────────────────────────────────────────────────────
def esm_fold(aa_seq: str, timeout: int = 30) -> Optional[dict]:
    """Submit sequence to ESMFold API. Returns dict with pLDDT or None on failure."""
    try:
        import requests
        url = "https://api.esmatlas.com/foldSequence/v1/pdb/"
        resp = requests.post(url, data=aa_seq, timeout=timeout,
                             headers={"Content-Type": "application/x-www-form-urlencoded"})
        if resp.status_code != 200:
            return None
        pdb_text = resp.text
        # Extract pLDDT from bfactor column of ATOM records
        plddt_values = []
        for line in pdb_text.splitlines():
            if line.startswith("ATOM") and " CA " in line:
                try:
                    plddt_values.append(float(line[60:66].strip()))
                except ValueError:
                    pass
        if not plddt_values:
            return None
        return {
            "plddt_mean": float(np.mean(plddt_values)),
            "plddt_min": float(np.min(plddt_values)),
            "plddt_max": float(np.max(plddt_values)),
            "pdb_text": pdb_text,
        }
    except Exception as exc:
        print(f"  [ESMFold] error: {exc}")
        return None


# ────────────────────────────────────────────────────────────────────────────
# Main loop
# ────────────────────────────────────────────────────────────────────────────
def run_design_loop(
    run_dir: str,
    critic_ckpt: str,
    critic_cfg: str,
    n_sequences: int = 50,
    max_attempts: int = 20,
    max_codons: int = 300,
    min_aa_length: int = 50,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.0,
    anneal_temp: bool = False,
    # T1a: critic-guided stability filter
    min_stability: float = 0.0,
    max_stability_attempts: int = 10,
    # T1b: family targeting
    target_family_idx: int = -1,
    min_family_conf: float = 0.3,
    esm_fold_top: int = 0,
    out_dir: str = "outputs/reports/generative_design",
) -> dict:
    device = dev()
    print(f"[design] device={device}")

    # Load models
    print("[design] Loading CodonLM...")
    codon_model, itos, stoi = load_codon_lm(run_dir, device)
    n_params = sum(p.numel() for p in codon_model.parameters())
    print(f"[design] CodonLM loaded: {n_params:,} params")

    print("[design] Loading ProteinCritic...")
    critic_model, tokenizer, task_dims = load_critic(critic_ckpt, critic_cfg, device)
    print(f"[design] Critic loaded: tasks={list(task_dims.keys())}")

    # Create output directory early so ESMFold can write PDB files during the loop
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    use_stability_filter = min_stability > 0.0
    use_family_filter = target_family_idx >= 0
    filter_desc = []
    if use_stability_filter:
        filter_desc.append(f"min_stability={min_stability}")
    if use_family_filter:
        filter_desc.append(f"target_family={target_family_idx}(conf>{min_family_conf})")
    if anneal_temp:
        filter_desc.append("anneal_temp")
    if top_p > 0:
        filter_desc.append(f"top_p={top_p}")
    if filter_desc:
        print(f"[design] Active filters: {', '.join(filter_desc)}")

    print(f"\n[design] Generating {n_sequences} sequences "
          f"(max_attempts={max_attempts}, max_stability_attempts={max_stability_attempts})...")
    records: list[dict] = []
    total_attempts = 0
    terminated_count = 0
    t_start = time.time()

    for seq_idx in range(n_sequences):
        # Outer critic-guided ReD loop (T1a + T1b)
        # Inner red_generate handles termination + length.
        # Outer loop rejects on stability / family — tries up to max_stability_attempts.
        best_rec: dict | None = None
        outer_total_att = 0

        for _outer in range(max_stability_attempts if (use_stability_filter or use_family_filter) else 1):
            codons, terminated, n_att = red_generate(
                codon_model, device, stoi, itos,
                max_codons=max_codons, max_attempts=max_attempts,
                temperature=temperature, top_k=top_k, top_p=top_p,
                min_aa_length=min_aa_length, anneal_temp=anneal_temp,
            )
            outer_total_att += n_att

            dna_seq = "".join(codons)
            aa_seq, aa_terminated = translate_dna(dna_seq)

            rec: dict = {
                "seq_id": seq_idx,
                "dna_seq": dna_seq,
                "aa_seq": aa_seq,
                "n_codons": len(codons),
                "n_aa": len(aa_seq),
                "terminated": terminated,
                "aa_terminated": aa_terminated,
                "n_attempts": outer_total_att,
            }

            # Always score with critic (needed for filter checks below)
            if aa_seq:
                crit = score_with_critic(critic_model, tokenizer, task_dims, aa_seq, device)
                rec.update(crit)

            # Keep best-so-far as fallback (highest stability)
            if best_rec is None or rec.get("stability_prob", 0) > best_rec.get("stability_prob", 0):
                best_rec = rec

            # T1a: stability filter
            if use_stability_filter and rec.get("stability_prob", 0) < min_stability:
                continue  # discard, retry

            # T1b: family targeting
            if use_family_filter:
                if (rec.get("family_top1", -1) != target_family_idx or
                        rec.get("family_top1_conf", 0) < min_family_conf):
                    continue  # discard, retry

            # All filters passed
            best_rec = rec
            break

        total_attempts += best_rec["n_attempts"]
        if best_rec.get("terminated", False):
            terminated_count += 1
        records.append(best_rec)

        if (seq_idx + 1) % 10 == 0:
            elapsed = time.time() - t_start
            stab_mean = np.mean([r.get("stability_prob", 0) for r in records if "stability_prob" in r])
            print(f"  {seq_idx+1}/{n_sequences} done  |  "
                  f"terminated={terminated_count}/{seq_idx+1}  |  "
                  f"stability_mean={stab_mean:.3f}  |  "
                  f"{elapsed:.1f}s elapsed")

    elapsed_total = time.time() - t_start
    termination_rate = terminated_count / n_sequences

    # Diversity metrics
    aa_seqs = [r["aa_seq"] for r in records if r["aa_seq"]]
    codon_seqs = [list(r["dna_seq"][i:i+3] for i in range(0, len(r["dna_seq"]), 3))
                  for r in records]
    pairwise_id = pairwise_identity(aa_seqs)
    kmer_div = kmer_diversity(aa_seqs, k=3)
    gc_vals = gc_content(codon_seqs)
    gc_mean = float(np.mean(gc_vals)) if gc_vals else 0.0
    gc_std = float(np.std(gc_vals)) if gc_vals else 0.0
    lengths = [r["n_aa"] for r in records]

    # Stability stats
    stab_probs = [r.get("stability_prob", 0.0) for r in records if "stability_prob" in r]
    fam_confs = [r.get("family_top1_conf", 0.0) for r in records if "family_top1_conf" in r]
    fn_confs = [r.get("function_top1_conf", 0.0) for r in records if "function_top1_conf" in r]

    # ESMFold for top sequences (by stability_prob)
    esm_results = []
    if esm_fold_top > 0:
        top_by_stability = sorted(
            [r for r in records if "stability_prob" in r and r["aa_seq"]],
            key=lambda x: x["stability_prob"], reverse=True
        )[:esm_fold_top]
        print(f"\n[design] Submitting top {len(top_by_stability)} sequences to ESMFold API...")
        for rank, rec in enumerate(top_by_stability, 1):
            print(f"  Seq {rec['seq_id']} (stability={rec.get('stability_prob', 0):.3f}, "
                  f"len={rec['n_aa']})...")
            fold = esm_fold(rec["aa_seq"])
            esm_result = {"seq_id": rec["seq_id"], "rank": rank}
            if fold:
                esm_result.update({
                    "plddt_mean": fold["plddt_mean"],
                    "plddt_min": fold["plddt_min"],
                    "plddt_max": fold["plddt_max"],
                })
                # Save PDB
                pdb_path = Path(out_dir) / f"top_{rank}_seq{rec['seq_id']}.pdb"
                pdb_path.write_text(fold["pdb_text"])
                esm_result["pdb_path"] = str(pdb_path)
                print(f"    pLDDT={fold['plddt_mean']:.1f}  →  saved {pdb_path.name}")
            else:
                print(f"    ESMFold failed or unavailable")
            esm_results.append(esm_result)

    # Save outputs  (out_path already created at function start)

    # CSV
    csv_path = out_path / "design_library.csv"
    fieldnames = list(records[0].keys()) if records else []
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            # Flatten lists for CSV
            flat = {k: (json.dumps(v) if isinstance(v, list) else v)
                    for k, v in rec.items()}
            writer.writerow(flat)
    print(f"\n[design] Saved library → {csv_path}")

    # ESMFold CSV
    if esm_results:
        esm_csv = out_path / "esm_fold_results.csv"
        with open(esm_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(esm_results[0].keys()), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(esm_results)
        print(f"[design] Saved ESMFold results → {esm_csv}")

    # Markdown report
    report = _build_report(
        records=records,
        n_sequences=n_sequences,
        termination_rate=termination_rate,
        total_attempts=total_attempts,
        elapsed_total=elapsed_total,
        task_dims=task_dims,
        pairwise_id=pairwise_id,
        kmer_div=kmer_div,
        gc_mean=gc_mean,
        gc_std=gc_std,
        lengths=lengths,
        stab_probs=stab_probs,
        fam_confs=fam_confs,
        fn_confs=fn_confs,
        esm_results=esm_results,
        max_attempts=max_attempts,
        temperature=temperature,
        top_k=top_k,
    )
    report_path = out_path / "design_report.md"
    report_path.write_text(report)
    print(f"[design] Saved report → {report_path}")

    return {
        "n_sequences": n_sequences,
        "terminated_count": terminated_count,
        "termination_rate": termination_rate,
        "pairwise_identity": pairwise_id,
        "kmer_diversity": kmer_div,
        "gc_mean": gc_mean,
        "stability_mean": float(np.mean(stab_probs)) if stab_probs else None,
        "csv_path": str(csv_path),
        "report_path": str(report_path),
    }


def _build_report(
    records, n_sequences, termination_rate, total_attempts, elapsed_total,
    task_dims, pairwise_id, kmer_div, gc_mean, gc_std, lengths,
    stab_probs, fam_confs, fn_confs, esm_results, max_attempts, temperature, top_k
) -> str:
    lines = [
        "# Generative Design Loop — Report",
        "",
        f"**Generated:** {n_sequences} sequences  |  "
        f"**Temperature:** {temperature}  |  **Top-K:** {top_k}  |  **Max-attempts:** {max_attempts}",
        f"**Elapsed:** {elapsed_total:.1f}s  |  "
        f"**Total CodonLM forward passes:** {total_attempts}",
        "",
        "---",
        "",
        "## 1. Termination (ReD Sampling)",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Sequences requested | {n_sequences} |",
        f"| Properly terminated | {int(termination_rate * n_sequences)} "
        f"({termination_rate*100:.1f}%) |",
        f"| Avg. attempts per sequence | {total_attempts/n_sequences:.2f} |",
        f"| Max attempts allowed | {max_attempts} |",
        "",
        "> **ReD interpretation:** Termination rate measures how often the model "
        "produces a valid stop codon within the max-codons budget. "
        "Higher temperature → lower termination rate; lower temperature → shorter, more stereotyped sequences.",
        "",
        "---",
        "",
        "## 2. Sequence Statistics",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Mean AA length | {np.mean(lengths):.1f} ± {np.std(lengths):.1f} |",
        f"| Min / Max AA length | {np.min(lengths)} / {np.max(lengths)} |",
        f"| Mean GC content | {gc_mean*100:.1f}% ± {gc_std*100:.1f}% |",
        "",
        "---",
        "",
        "## 3. ProteinCritic Scores",
        "",
    ]

    if stab_probs:
        high_stab = sum(1 for p in stab_probs if p > 0.7)
        lines += [
            f"### Stability",
            f"| Metric | Value |",
            f"|---|---|",
            f"| Mean stability probability | {np.mean(stab_probs):.3f} |",
            f"| Sequences with P(stable) > 0.7 | {high_stab} / {len(stab_probs)} "
            f"({high_stab/len(stab_probs)*100:.1f}%) |",
            f"| Sequences with P(stable) > 0.9 | "
            f"{sum(1 for p in stab_probs if p > 0.9)} / {len(stab_probs)} |",
            "",
        ]

    if fam_confs:
        lines += [
            f"### Pfam Family",
            f"| Metric | Value |",
            f"|---|---|",
            f"| Mean top-1 confidence | {np.mean(fam_confs):.4f} |",
            f"| Sequences with conf > 0.5 | "
            f"{sum(1 for c in fam_confs if c > 0.5)} / {len(fam_confs)} |",
            "",
        ]

    if fn_confs:
        lines += [
            f"### EC Function",
            f"| Metric | Value |",
            f"|---|---|",
            f"| Mean top-1 confidence | {np.mean(fn_confs):.4f} |",
            f"| Sequences with conf > 0.5 | "
            f"{sum(1 for c in fn_confs if c > 0.5)} / {len(fn_confs)} |",
            "",
        ]

    lines += [
        "---",
        "",
        "## 4. Diversity",
        "",
        f"| Metric | Value | Interpretation |",
        f"|---|---|---|",
        f"| Mean pairwise AA identity | {pairwise_id*100:.1f}% | "
        f"{'Diverse (< 70%)' if pairwise_id < 0.7 else 'Redundant (> 70%)'} |",
        f"| 3-mer AA k-mer coverage | {kmer_div*100:.2f}% of possible space | "
        f"{'Broad' if kmer_div > 0.1 else 'Narrow'} |",
        "",
        "> **Natural proteins** typically share < 30% identity across different families. "
        f"{'✅ Generated library shows good diversity.' if pairwise_id < 0.5 else '⚠️ Generated library is relatively homogeneous — try higher temperature.'}",
        "",
        "---",
        "",
    ]

    # ESMFold section
    if esm_results:
        lines += [
            "## 5. ESMFold Structure Predictions (Top Sequences)",
            "",
            f"| Rank | Seq ID | pLDDT (mean) | pLDDT (min) | Confidence |",
            f"|---|---|---|---|---|",
        ]
        for r in esm_results:
            plddt = r.get("plddt_mean", "N/A")
            plddt_min = r.get("plddt_min", "N/A")
            if isinstance(plddt, float):
                # ESMFold API returns pLDDT on 0–1 scale
                conf = ("🟢 High (>0.9)" if plddt > 0.9 else
                        "🟡 Moderate (0.7–0.9)" if plddt > 0.7 else
                        "🔴 Low (<0.7) — likely disordered")
                lines.append(f"| {r['rank']} | {r['seq_id']} | {plddt:.3f} | "
                              f"{plddt_min:.3f} | {conf} |")
            else:
                lines.append(f"| {r['rank']} | {r['seq_id']} | N/A | N/A | Failed |")
        lines.append("")
        lines.append("> pLDDT > 0.7 indicates confident structure (ESMFold 0–1 scale). "
                     "Low pLDDT is expected for novel sequences outside training distribution. "
                     "PDB files saved alongside this report.")
        lines += ["", "---", ""]

    # Top 5 sequences by stability
    top_seqs = sorted(
        [r for r in records if "stability_prob" in r and r["aa_seq"]],
        key=lambda x: x["stability_prob"], reverse=True
    )[:5]
    if top_seqs:
        lines += [
            "## Top 5 Sequences by Stability",
            "",
            "| Seq ID | AA Length | Stability | Family Conf | Function Conf | AA sequence (first 60) |",
            "|---|---|---|---|---|---|",
        ]
        for r in top_seqs:
            aa_preview = r["aa_seq"][:60] + ("..." if len(r["aa_seq"]) > 60 else "")
            lines.append(
                f"| {r['seq_id']} | {r['n_aa']} | {r.get('stability_prob', 0):.3f} | "
                f"{r.get('family_top1_conf', 0):.4f} | {r.get('function_top1_conf', 0):.4f} | "
                f"`{aa_preview}` |"
            )
        lines.append("")

    return "\n".join(lines)


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Generative Design Loop")
    ap.add_argument(
        "--run_dir",
        default="runs/2026-06-15_stage2.6_10L8H_d384_e10",
        help="CodonLM run directory (contains checkpoints/best.pt)",
    )
    ap.add_argument(
        "--critic_ckpt",
        default="runs/protein_critic/checkpoints/best_critic.pt",
        help="Path to MultiTask ProteinCritic checkpoint",
    )
    ap.add_argument(
        "--critic_cfg",
        default="configs/protein_critic.yaml",
        help="Path to ProteinCritic YAML config",
    )
    ap.add_argument("--n_sequences", type=int, default=50)
    ap.add_argument("--max_attempts", type=int, default=20,
                    help="ReD max attempts per sequence before giving up")
    ap.add_argument("--max_codons", type=int, default=300)
    ap.add_argument("--min_aa_length", type=int, default=50,
                    help="Minimum AA length to accept; shorter sequences are discarded (ReD)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=50,
                    help="Top-K sampling (0 = disabled). Ignored when --top_p > 0.")
    ap.add_argument("--top_p", type=float, default=0.0,
                    help="Nucleus (top-p) sampling mass (0 = disabled, overrides --top_k)")
    ap.add_argument("--anneal_temp", action="store_true",
                    help="T1c: linearly anneal temperature to 0.7x after first 50 codons")
    # T1a: critic-guided stability filter
    ap.add_argument("--min_stability", type=float, default=0.0,
                    help="T1a: reject sequences with stability_prob < this (0 = disabled)")
    ap.add_argument("--max_stability_attempts", type=int, default=10,
                    help="T1a: max outer retries for the stability/family filter")
    # T1b: family targeting
    ap.add_argument("--target_family_idx", type=int, default=-1,
                    help="T1b: only accept sequences where family_top1 == this index (-1 = disabled)")
    ap.add_argument("--min_family_conf", type=float, default=0.3,
                    help="T1b: minimum family confidence required when --target_family_idx is set")
    ap.add_argument("--esm_fold_top", type=int, default=0,
                    help="Submit top-N sequences to ESMFold API (0 = disabled, requires internet)")
    ap.add_argument("--out_dir", default="outputs/reports/generative_design")
    args = ap.parse_args()

    result = run_design_loop(
        run_dir=args.run_dir,
        critic_ckpt=args.critic_ckpt,
        critic_cfg=args.critic_cfg,
        n_sequences=args.n_sequences,
        max_attempts=args.max_attempts,
        max_codons=args.max_codons,
        min_aa_length=args.min_aa_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        anneal_temp=args.anneal_temp,
        min_stability=args.min_stability,
        max_stability_attempts=args.max_stability_attempts,
        target_family_idx=args.target_family_idx,
        min_family_conf=args.min_family_conf,
        esm_fold_top=args.esm_fold_top,
        out_dir=args.out_dir,
    )

    print("\n=== Design Loop Summary ===")
    for k, v in result.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
