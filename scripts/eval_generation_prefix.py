#!/usr/bin/env python3
"""
Prefix-generation benchmark for codon LM.

Given a run_id and a list of prefix lengths (k in codons), generate continuations
from real CDS prefixes, score validity/fidelity/coherence metrics, and produce:

- outputs/scores/<RUN_ID>/gen_prefix/samples.csv
- outputs/scores/<RUN_ID>/gen_prefix/summary.csv
- simple plots under outputs/scores/<RUN_ID>/gen_prefix/

CLI:
  python -m scripts.eval_generation_prefix --run_id <RUN_ID> \
    --k_list 1,3,5,10 --samples 5 --max_genes 50 --max_new 300 --temperature 0.8 --topk 5
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from . import query_model as Q
from src.codonlm.generate import generate_cds_constrained


# --- Biology helpers ---
CODON_TO_AA: Dict[str, str] = {
    # fmt: off
    "TTT":"F","TTC":"F","TTA":"L","TTG":"L","TCT":"S","TCC":"S","TCA":"S","TCG":"S",
    "TAT":"Y","TAC":"Y","TAA":"Stop","TAG":"Stop","TGT":"C","TGC":"C","TGA":"Stop","TGG":"W",
    "CTT":"L","CTC":"L","CTA":"L","CTG":"L","CCT":"P","CCC":"P","CCA":"P","CCG":"P",
    "CAT":"H","CAC":"H","CAA":"Q","CAG":"Q","CGT":"R","CGC":"R","CGA":"R","CGG":"R",
    "ATT":"I","ATC":"I","ATA":"I","ATG":"M","ACT":"T","ACC":"T","ACA":"T","ACG":"T",
    "AAT":"N","AAC":"N","AAA":"K","AAG":"K","AGT":"S","AGC":"S","AGA":"R","AGG":"R",
    "GTT":"V","GTC":"V","GTA":"V","GTG":"V","GCT":"A","GCC":"A","GCA":"A","GCG":"A",
    "GAT":"D","GAC":"D","GAA":"E","GAG":"E","GGT":"G","GGC":"G","GGA":"G","GGG":"G",
    # fmt: on
}


def _codon_to_aa(codon: str) -> str:
    return CODON_TO_AA.get(codon, "?")


def _aa_seq(codons: List[str]) -> List[str]:
    return [_codon_to_aa(c) for c in codons if len(c) == 3]


def _ngram_repeat_ratio(tokens: List[str], n: int = 3) -> float:
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    uniq = len(set(grams))
    total = len(grams)
    return 1.0 - (uniq / total) if total else 0.0


def _score_stop_behavior(gen_codons: List[str], truth_len_codons: int) -> Tuple[float, bool, bool]:
    """Return (StopScore, valid_end_stop, early_stop_flag).

    StopScore = 1 if ends with canonical stop and <eog> present; else decays with normalized
    termination error.
    """
    stops = {"TAA", "TAG", "TGA"}
    valid_end = len(gen_codons) > 0 and gen_codons[-1] in stops
    # early stop: any stop before 90% of truth length
    early = False
    cutoff = max(1, int(0.9 * truth_len_codons))
    for i, c in enumerate(gen_codons[:-1]):
        if c in stops and i < cutoff:
            early = True
            break
    if valid_end:
        stop_score = 1.0 if not early else 0.5
    else:
        # termination error: absolute distance from truth length (normalized)
        tau = abs(len(gen_codons) - truth_len_codons) / max(1, truth_len_codons)
        stop_score = max(0.0, 1.0 - tau / 0.2)  # decay to 0 after ~20% error
    return float(stop_score), bool(valid_end), bool(early)


@dataclass
class SampleResult:
    run_id: str
    gene_idx: int
    k: int
    sample_id: int
    aa_identity: float
    syn_rate: float
    stop_score: float
    frame_integrity: float
    ppl_stability: float
    no_repeat: float
    usage_agree: float
    gqs: float
    gen_len: int
    valid_end: bool
    early_stop: bool
    # long-protein generation metadata
    gen_len_codons: int
    had_terminal_stop: bool
    hit_hard_cap: bool
    target_codons: int


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--k_list", default="1,3,5,10")
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--max_genes", type=int, default=50)
    ap.add_argument("--max_new", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--topk", type=int, default=5)
    # Long-protein controls
    ap.add_argument("--min_aa_len", type=int, default=100)
    ap.add_argument("--target_aa_len", type=int, default=256)
    ap.add_argument("--max_aa_len", type=int, default=400)
    ap.add_argument("--require_terminal_stop", action="store_true", default=False)
    ap.add_argument("--special_margin", type=int, default=6)
    # Normalization option for GQS
    ap.add_argument("--gqs_normalize", choices=["none","len"], default="none",
                    help="Normalize GQS by reference length (truth length if available, else gen length)")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    run_dir = repo / "runs" / args.run_id
    out_dir = repo / "outputs" / "scores" / args.run_id / "gen_prefix"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load tokens + checkpoint
    itos, stoi = Q._load_vocab(run_dir)
    state_dict, cfg = Q._load_checkpoint(run_dir)
    model = Q.build_model_from_state(state_dict, cfg)
    device = Q.dev()
    model.to(device).eval()
    # Validate AA length constraints
    if not (0 < args.min_aa_len <= args.target_aa_len <= args.max_aa_len):
        raise SystemExit("require 0 < min_aa_len ≤ target_aa_len ≤ max_aa_len")

    # Choose CDS corpus: from first dataset dna path in combined manifest
    dna_path = None
    manifest = run_dir / "combined_manifest.json"
    if manifest.exists():
        data = json.loads(manifest.read_text())
        if data.get("datasets"):
            dna_path = Path(data["datasets"][0]["dna"])
            if not dna_path.is_absolute():
                dna_path = repo / dna_path
    if dna_path is None or not dna_path.exists():
        raise SystemExit("[gen-prefix] could not locate a CDS dna file via combined_manifest.json")

    # Build reference corpus codon unigram from test manifest if present (fallback: from dna file)
    codon_mask = np.array([1 if (len(t) == 3 and all(c in "ACGT" for c in t)) else 0 for t in itos], dtype=np.int32)
    unigram = np.zeros((len(itos),), dtype=np.float64)
    # quick pass over dna file
    lines = 0
    with open(dna_path) as f:
        for line in f:
            seq = line.strip().upper().replace("U","T")
            L = (len(seq) // 3) * 3
            for i in range(0, L, 3):
                tok = seq[i:i+3]
                j = stoi.get(tok, None)
                if j is not None:
                    unigram[j] += 1
            lines += 1
            if lines >= 10000:
                break
    if unigram.sum() == 0:
        unigram[:] = 1.0
    unigram_cod = unigram * codon_mask
    unigram_cod = unigram_cod / max(1e-9, unigram_cod.sum())

    # Load CDS list
    cds: List[str] = []
    with open(dna_path) as f:
        for line in f:
            s = line.strip().upper().replace("U","T")
            if len(s) >= 9:
                cds.append(s)
            if len(cds) >= args.max_genes:
                break

    def aa_identity(truth: List[str], gen: List[str]) -> float:
        L = min(len(truth), len(gen))
        if L == 0:
            return 0.0
        return float(sum(1 for i in range(L) if truth[i] == gen[i])) / L

    def syn_rate(truth_cod: List[str], gen_cod: List[str]) -> float:
        L = min(len(truth_cod), len(gen_cod))
        if L == 0:
            return 0.0
        cnt = 0
        for i in range(L):
            a, b = _codon_to_aa(truth_cod[i]), _codon_to_aa(gen_cod[i])
            if a != "Stop" and b != "Stop" and a == b:
                cnt += 1
        return float(cnt) / L

    def ppl_stability(ids: List[int]) -> float:
        # Use mean NLL of first/last 10 tokens of continuation (approximate drift)
        if len(ids) < 22:
            return 1.0
        x = torch.tensor(ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
        y = torch.tensor(ids[1:], dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = model(x, y)
            loss_all = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=0, reduction="none"
            ).view(1, -1)
        # first/last window
        w = min(10, loss_all.shape[1] // 4)
        first = loss_all[0, :w].mean().item()
        last = loss_all[0, -w:].mean().item()
        s = max(0.0, last - first)
        # map slope to [0,1]
        return float(np.exp(-s / 0.02))

    def usage_agree(gen_ids: List[int]) -> float:
        counts = np.zeros_like(unigram)
        for j in gen_ids:
            counts[int(j)] += 1
        p = counts * codon_mask
        s = p.sum()
        if s <= 0:
            return 0.0
        p = p / s
        kl = float((p * (np.log((p + 1e-12) / (unigram_cod + 1e-12)))).sum())
        # scale to [0,1] with heuristic KL0
        KL0 = 0.5
        return float(max(0.0, 1.0 - min(1.0, kl / KL0)))

    def frame_integrity_ok(gen_codons: List[str]) -> float:
        ok = all(len(c) == 3 and set(c) <= set("ACGT") for c in gen_codons)
        return 1.0 if ok else 0.0

    def gqs(stop_score, aaid, syn, stab, norep, usage, frame) -> float:
        return 100.0 * (0.30 * stop_score + 0.20 * aaid + 0.15 * syn + 0.10 * stab + 0.10 * norep + 0.10 * usage + 0.05 * frame)

    rows: List[SampleResult] = []
    k_list = [int(x) for x in args.k_list.split(",") if x]

    block_size = int(cfg.get("block_size", getattr(model, "block_size", 512)))
    for gene_idx, dna in enumerate(cds):
        truth_codons = [dna[i:i+3] for i in range(0, (len(dna)//3)*3, 3)]
        truth_aa = _aa_seq(truth_codons)
        for k in k_list:
            prefix = dna[: 3 * min(k, len(truth_codons))]
            # tokenize prefix
            ctx_ids = Q.dna_to_ids(prefix, stoi)
            for sidx in range(args.samples):
                # Compute safe generation lengths (AA == codons)
                max_window_codons = block_size - int(k) - int(args.special_margin)
                if max_window_codons < args.min_aa_len:
                    raise ValueError("block_size too small for requested lengths and k")
                hard_cap = int(min(max_window_codons, args.max_aa_len, args.max_new))
                target_codons = int(min(args.target_aa_len, hard_cap))
                target_codons = int(max(target_codons, args.min_aa_len))
                # Constrained generation
                gen_ids, info = generate_cds_constrained(
                    model=model,
                    device=device,
                    ctx_ids=ctx_ids,
                    stoi=stoi,
                    itos=itos,
                    target_codons=target_codons,
                    hard_cap=hard_cap,
                    require_terminal_stop=bool(args.require_terminal_stop),
                    temperature=float(args.temperature),
                    topk=int(args.topk) if args.topk>0 else 0,
                )
                gen_toks = Q.ids_to_codons(gen_ids, itos)
                # strip BOS and anything before first codon
                codons = [t for t in gen_toks if len(t) == 3 and set(t) <= set("ACGT")]
                # continuation after prefix length
                gen_cont_cod = codons[min(k, len(codons)) :]
                gen_cont_ids = [stoi[c] for c in gen_cont_cod if c in stoi]
                gen_cont_aa = _aa_seq(gen_cont_cod)
                # metrics
                aaid = aa_identity(truth_aa[k:], gen_cont_aa)
                syn = syn_rate(truth_codons[k:], gen_cont_cod)
                stop_score, valid_end, early = _score_stop_behavior(codons, truth_len_codons=len(truth_codons))
                stab = ppl_stability([stoi.get(c, 0) for c in codons])
                norep = 1.0 - _ngram_repeat_ratio(codons, n=3)
                usage = usage_agree(gen_cont_ids)
                frame = frame_integrity_ok(codons)
                score = gqs(stop_score, aaid, syn, stab, norep, usage, frame)
                rows.append(SampleResult(
                    args.run_id, gene_idx, k, sidx, aaid, syn, stop_score, frame, stab, norep, usage, score,
                    len(codons), valid_end, early,
                    gen_len_codons=len(codons), had_terminal_stop=bool(info.get("had_terminal_stop", False)),
                    hit_hard_cap=bool(info.get("hit_hard_cap", False)), target_codons=int(target_codons)
                ))

    # write samples.csv
    samples_csv = out_dir / "samples.csv"
    with samples_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([c for c in SampleResult.__annotations__.keys()])
        for r in rows:
            writer.writerow([getattr(r, c) for c in SampleResult.__annotations__.keys()])
    print(f"[gen-prefix] wrote {samples_csv}")

    # summary per k
    import statistics as stats
    summary = []
    for k in k_list:
        rks = [r for r in rows if r.k == k]
        if not rks:
            continue
        term_rate = sum(1 for r in rks if r.valid_end) / len(rks)
        early_rate = sum(1 for r in rks if r.early_stop) / len(rks)
        median_gqs = float(stats.median([r.gqs for r in rks]))
        mean_aa = float(sum(r.aa_identity for r in rks) / len(rks))
        best_aa = float(max(r.aa_identity for r in rks))
        mean_len = float(sum(r.gen_len_codons for r in rks) / len(rks))
        median_len = float(stats.median([r.gen_len_codons for r in rks]))
        stop_rate = sum(1 for r in rks if r.had_terminal_stop) / len(rks)
        hard_cap_rate = sum(1 for r in rks if r.hit_hard_cap) / len(rks)
        # Optional length-normalized GQS
        mean_gqs_norm = None
        median_gqs_norm = None
        if args.gqs_normalize == "len":
            # Use target_codons (proxy for truth length) when available; fallback to gen_len
            norms = []
            for r in rks:
                denom = max(1, getattr(r, "target_codons", 0) or r.gen_len_codons)
                norms.append(r.gqs / float(denom))
            if norms:
                mean_gqs_norm = float(sum(norms) / len(norms))
                median_gqs_norm = float(stats.median(norms))
        summary.append({
            "k": k,
            "termination_rate": term_rate,
            "early_stop_rate": early_rate,
            "median_gqs": median_gqs,
            "mean_aa_identity": mean_aa,
            "best_aa_identity": best_aa,
            "mean_aa_len": mean_len,
            "median_aa_len": median_len,
            "terminal_stop_rate": stop_rate,
            "hard_cap_rate": hard_cap_rate,
            **({"mean_gqs_norm": mean_gqs_norm, "median_gqs_norm": median_gqs_norm} if args.gqs_normalize == "len" else {}),
            "n": len(rks),
        })
    summary_csv = out_dir / "summary.csv"
    with summary_csv.open("w", newline="") as f:
        base_cols = [
            "k","termination_rate","early_stop_rate","median_gqs","mean_aa_identity","best_aa_identity",
            "mean_aa_len","median_aa_len","terminal_stop_rate","hard_cap_rate","n"
        ]
        extra = ["mean_gqs_norm","median_gqs_norm"] if any("mean_gqs_norm" in s for s in summary) else []
        writer = csv.DictWriter(f, fieldnames=base_cols + extra)
        writer.writeheader(); writer.writerows(summary)
    print(f"[gen-prefix] wrote {summary_csv}")

    # simple plots
    try:
        ks = [s["k"] for s in summary]
        tr = [s["termination_rate"] for s in summary]
        gq = [s["median_gqs"] for s in summary]
        aa = [s["mean_aa_identity"] for s in summary]
        plt.figure(); plt.plot(ks, tr, marker='o'); plt.xlabel('k'); plt.ylabel('termination_rate'); plt.title('Termination vs k'); plt.tight_layout(); plt.savefig(out_dir/"termination_vs_k.png"); plt.close()
        plt.figure(); plt.plot(ks, gq, marker='o'); plt.xlabel('k'); plt.ylabel('median_gqs'); plt.title('GQS vs k'); plt.tight_layout(); plt.savefig(out_dir/"gqs_vs_k.png"); plt.close()
        plt.figure(); plt.plot(ks, aa, marker='o'); plt.xlabel('k'); plt.ylabel('mean_aa_identity'); plt.title('AA identity vs k'); plt.tight_layout(); plt.savefig(out_dir/"aa_vs_k.png"); plt.close()
        ml = [s["mean_aa_len"] for s in summary]
        plt.figure(); plt.plot(ks, ml, marker='o'); plt.xlabel('k'); plt.ylabel('mean_aa_len'); plt.title('AA length vs k'); plt.tight_layout();
        plt.savefig(out_dir/"aa_len_vs_k.png");
        try:
            figs_root = Path(__file__).resolve().parents[1] / "outputs" / "figs"
            figs_root.mkdir(parents=True, exist_ok=True)
            plt.savefig(figs_root/"aa_len_vs_k.png")
        except Exception:
            pass
        plt.close()
    except Exception as exc:
        print(f"[gen-prefix] plotting failed: {exc}")


if __name__ == "__main__":
    main()
