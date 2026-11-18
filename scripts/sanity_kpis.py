#!/usr/bin/env python3
"""
Compute sanity KPIs on a held-out subset of test windows and merge them into metrics.json.

KPIs:
- codon_corr: Pearson correlation between model unigram codon distribution (aggregated next-token probs) and corpus codon frequencies.
- frameshift_delta: mean ΔNLL when shifting the input window by +1 token (codon).
- startstop_delta.start / .stop: mean ΔNLL when mutating a start (ATG) / stop (TAA/TAG/TGA) token in-context.
- syn_gap: average (ΔNLL_nonSyn − ΔNLL_syn).
- [optional] esm2_critic_delta_median: median ESM-2 log-likelihood gap (Real − Generated). Enable with --with_esm2.

Keeps runtime small by sampling at most N windows.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from src.codonlm.checkpoints import build_codon_model_from_cfg, load_codon_checkpoint
from src.codonlm.data_loading import PackedDataset, dynamic_lm_collate_fn
from src.codonlm.model_tiny_gpt import TinyGPT
from src.codonlm.metrics_io import write_merge_metrics
from scripts._shared import resolve_run


def dev() -> torch.device:
    import os
    if os.environ.get("FORCE_CPU") == "1":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


CODON_TO_AA: Dict[str, str] = {
    # fmt: off
    "TTT": "F",
    "TTC": "F",
    "TTA": "L",
    "TTG": "L",
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "TAT": "Y",
    "TAC": "Y",
    "TAA": "Stop",
    "TAG": "Stop",
    "TGT": "C",
    "TGC": "C",
    "TGA": "Stop",
    "TGG": "W",
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "CAT": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "ATT": "I",
    "ATC": "I",
    "ATA": "I",
    "ATG": "M",
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "AAT": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "AGT": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "GAT": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
    # fmt: on
}


def _load_checkpoint(run_dir: Path) -> tuple[dict, dict]:
    state_dict, cfg, _ = load_codon_checkpoint(run_dir)
    return state_dict, cfg


def _build_model_from_cfg(cfg: dict) -> TinyGPT:
    return build_codon_model_from_cfg(cfg)


def load_vocab(run_id: str, repo_root: Path) -> Tuple[List[str], Dict[str, int]]:
    itos_path = repo_root / "runs" / run_id / "itos.txt"
    tokens = [
        line.strip() for line in itos_path.read_text().splitlines() if line.strip()
    ]
    stoi = {tok: i for i, tok in enumerate(tokens)}
    return tokens, stoi


@torch.no_grad()
def batch_nll(
    model: TinyGPT, device: torch.device, xb: torch.Tensor, yb: torch.Tensor
) -> float:
    xb = xb.to(device)
    yb = yb.to(device)
    logits, loss = model(xb, yb)
    if loss is None:
        return 0.0
    valid = (yb != 0).sum().item()
    return float(loss.item()) * max(1, valid)


def codon_mask(itos: List[str]) -> np.ndarray:
    return np.array(
        [1 if (len(tok) == 3 and all(c in "ACGT" for c in tok)) else 0 for tok in itos],
        dtype=np.int32,
    )


def compute_kpis(
    model: TinyGPT,
    device: torch.device,
    test_npz: Path,
    itos: List[str],
    sample_windows: int = 200,
) -> Dict[str, float]:
    ds = PackedDataset(test_npz)
    n = min(sample_windows, len(ds))
    idxs = np.linspace(0, len(ds) - 1, num=n, dtype=int)
    if getattr(ds, "is_dynamic", False):
        x, y = dynamic_lm_collate_fn([ds[i] for i in idxs])
    else:
        x = ds.X[idxs]
        y = ds.Y[idxs]

    # 1) codon_corr
    cmask = codon_mask(itos)
    with torch.no_grad():
        xb = x.to(device)
        logits, _ = model(xb)
        probs = torch.softmax(logits, dim=-1)  # (B,T,V)
        model_unigram = probs.sum(dim=(0, 1)).cpu().numpy()  # (V,)
    # test corpus frequencies
    counts = np.zeros((len(itos),), dtype=np.float64)
    flat = x.view(-1).cpu().numpy()
    for v in flat:
        counts[int(v)] += 1
    # restrict to codons only
    codon_idx = np.where(cmask == 1)[0]
    a = model_unigram[codon_idx]
    b = counts[codon_idx]
    # normalize
    a = a / max(1e-9, a.sum())
    b = b / max(1e-9, b.sum())
    # Pearson correlation
    a0 = a - a.mean()
    b0 = b - b.mean()
    denom = (np.linalg.norm(a0) * np.linalg.norm(b0)) or 1.0
    codon_corr = float(np.dot(a0, b0) / denom)

    # 2) frameshift_delta (shift by +1 token)
    nll_orig = batch_nll(model, device, x, y)
    x_shift = x[:, 1:]
    y_shift = y[:, 1:]
    # clip to same length by trimming to min length
    minT = min(x.shape[1], x_shift.shape[1])
    nll_shift = batch_nll(model, device, x_shift[:, :minT], y_shift[:, :minT])
    frameshift_delta = float((nll_shift - nll_orig) / max(1, (y != 0).sum().item()))

    # 3) start/stop deltas
    stoi = {tok: i for i, tok in enumerate(itos)}
    start_idx = stoi.get("ATG", -1)
    stop_set = {stoi.get(s, -1) for s in ("TAA", "TAG", "TGA")}
    stop_set.discard(-1)
    codon_ids = [i for i, t in enumerate(itos) if len(t) == 3 and t[0] in "ACGT"]
    non_start = [i for i in codon_ids if i != start_idx]
    non_stop = [i for i in codon_ids if i not in stop_set]

    def mutate_once(xb: torch.Tensor, target_tok: int, pool: List[int]) -> float:
        # find first occurrence per row; mutate and measure ΔNLL
        xb_mut = xb.clone()
        changed = 0
        for i in range(xb.shape[0]):
            row = xb[i]
            pos = (row == target_tok).nonzero(as_tuple=True)[0]
            if pos.numel() == 0:
                continue
            p = int(pos[0].item())
            xb_mut[i, p] = int(random.choice(pool))
            changed += 1
        if changed == 0:
            return float("nan")
        nll_m = batch_nll(model, device, xb_mut, y)
        return float((nll_m - nll_orig) / max(1, (y != 0).sum().item()))

    start_delta = (
        mutate_once(x.clone(), start_idx, non_start) if start_idx >= 0 else float("nan")
    )
    stop_delta = float("nan")
    if stop_set:
        # mutate if any stop present; pick a representative code to locate positions, then mutate to non-stop
        # We'll scan all three stops and mutate whichever appears first per row
        xb_mut = x.clone()
        changed = 0
        for i in range(x.shape[0]):
            row = x[i]
            # find any stop position
            pos_all = None
            for s in stop_set:
                pos = (row == s).nonzero(as_tuple=True)[0]
                if pos.numel() > 0:
                    pos_all = pos if pos_all is None else torch.cat([pos_all, pos])
            if pos_all is None or pos_all.numel() == 0:
                continue
            p = int(pos_all[0].item())
            xb_mut[i, p] = int(random.choice(non_stop))
            changed += 1
        if changed > 0:
            nll_m = batch_nll(model, device, xb_mut, y)
            stop_delta = float((nll_m - nll_orig) / max(1, (y != 0).sum().item()))

    # 4) syn_gap
    syn_delta, nonsyn_delta, n_used = 0.0, 0.0, 0
    for i in range(x.shape[0]):
        row = x[i].clone()
        for p in range(1, row.shape[0] - 1):  # avoid BOS/edge
            tok = int(row[p].item())
            tok_str = itos[tok]
            if len(tok_str) != 3:
                continue
            aa = CODON_TO_AA.get(tok_str)
            if not aa or aa == "Stop":
                continue
            # collect synonym ids
            syn_ids = [
                j
                for j, t in enumerate(itos)
                if t in CODON_TO_AA and CODON_TO_AA[t] == aa and t != tok_str
            ]
            nonsyn_ids = [
                j
                for j, t in enumerate(itos)
                if t in CODON_TO_AA
                and CODON_TO_AA[t] != aa
                and t not in ("TAA", "TAG", "TGA")
            ]
            if not syn_ids or not nonsyn_ids:
                continue
            # mutate and measure per-position delta
            row_syn = row.clone()
            row_syn[p] = int(random.choice(syn_ids))
            row_nsn = row.clone()
            row_nsn[p] = int(random.choice(nonsyn_ids))
            # batch both with same y context length adjust
            xb = torch.stack([row, row_syn, row_nsn], dim=0)
            batch_nll(
                model, device, xb, y[i : i + 1].repeat(3, 1)
            )  # returns sum over tokens; but our helper expects batch; adjust helper? reuse as is
            # Since batch_nll returns sum over valid tokens for entire batch, get average per example by dividing by 3 and token count is same across
            # To keep simple, compute deltas via separate calls
            nll0 = batch_nll(model, device, row.unsqueeze(0), y[i : i + 1])
            nllS = batch_nll(model, device, row_syn.unsqueeze(0), y[i : i + 1])
            nllN = batch_nll(model, device, row_nsn.unsqueeze(0), y[i : i + 1])
            syn_delta += float(nllS - nll0)
            nonsyn_delta += float(nllN - nll0)
            n_used += 1
            break  # one position per row
    syn_gap = float((nonsyn_delta - syn_delta) / max(1, n_used))

    return {
        "codon_corr": codon_corr,
        "frameshift_delta": frameshift_delta,
        "startstop_delta.start": start_delta,
        "startstop_delta.stop": stop_delta,
        "syn_gap": syn_gap,
    }


def compute_esm2_kpis(
    model: TinyGPT,
    device: torch.device,
    run_dir: Path,
    itos: List[str],
    stoi: Dict[str, int],
    *,
    num_seqs: int = 32,
    max_codons: int = 128,
    num_positions: int = 16,
    seed: int = 42,
    manifest: Path | None = None,
) -> Dict[str, float]:
    """Score num_seqs generated proteins vs length-matched real proteins with frozen ESM-2 (8M).

    Returns a dict with keys:
        esm2_critic_real_median_score
        esm2_critic_generated_median_score
        esm2_critic_delta_median
        esm2_critic_mann_whitney_p
    """
    from scripts.query_model import generate as cds_generate
    from src.eval.protein_critic_esm import (
        ESM2ProteinCritic,
        extract_length_matched_validation_proteins,
        run_esm2_critic_benchmark,
    )
    from src.eval.translation import translate_codons

    # Always use CPU for ESM-2 critic (sequential autoregressive decoding is faster on CPU for Mac)
    critic_device = torch.device("cpu")

    torch.manual_seed(seed)
    bos_id = stoi.get("<BOS_CDS>", 1)
    eos_id = stoi.get("<EOS_CDS>")

    generated_proteins: list[tuple[str, str]] = []
    target_lengths: list[int] = []

    gen_model = model
    gen_device = torch.device("cpu")
    gen_model_cpu = gen_model.to(gen_device) if device.type != "cpu" else gen_model

    for i in range(num_seqs):
        ids = cds_generate(
            gen_model_cpu,
            gen_device,
            ctx_ids=[bos_id],
            max_new=max_codons,
            temperature=1.0,
            topk=0,
            eos_idx=eos_id,
        )
        tokens = [itos[j] if 0 <= j < len(itos) else f"<{j}>" for j in ids]
        report = translate_codons(tokens)
        prot = report.protein_sequence or "M"
        generated_proteins.append((f"gen_{i:02d}", prot))
        target_lengths.append(len(prot))

    # Resolve manifest path
    repo = Path(__file__).resolve().parents[1]
    if manifest is None:
        manifest = repo / "data/processed/corrected/corrected-codonlm-v1/genome/manifest.json"

    real_proteins = extract_length_matched_validation_proteins(
        manifest,
        target_lengths,
        seed=seed,
    )

    critic = ESM2ProteinCritic(device=critic_device)
    summary = run_esm2_critic_benchmark(
        critic,
        generated_proteins,
        real_proteins,
        num_positions=num_positions,
        seed=seed,
    )

    # Persist full benchmark report alongside run scores
    scores_dir = run_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    out_dict = summary.to_dict()
    (scores_dir / "esm2_critic_benchmark.json").write_text(
        __import__("json").dumps(out_dict, indent=2) + "\n"
    )

    return {
        "esm2_critic_real_median_score": summary.median_real_score,
        "esm2_critic_generated_median_score": summary.median_generated_score,
        "esm2_critic_delta_median": summary.delta_median_score,
        "esm2_critic_mann_whitney_p": summary.p_value,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", help="outputs/checkpoints/<RUN_ID>")
    ap.add_argument("--run_id", help="Run id (alternative to --run_dir)")
    ap.add_argument("--max_windows", type=int, default=200)
    ap.add_argument("--test_npz", help="override test NPZ path")
    ap.add_argument(
        "--with_esm2",
        action="store_true",
        default=False,
        help="Also run the frozen ESM-2 (8M) protein critic KPI. "
             "Generates --esm2_num_seqs proteins, scores them vs length-matched "
             "real validation proteins, and merges results into metrics.json. "
             "Adds ~2 min on CPU for the default 32 sequences.",
    )
    ap.add_argument(
        "--esm2_num_seqs",
        type=int,
        default=32,
        help="Number of generated/real proteins to score with ESM-2 (default: 32).",
    )
    ap.add_argument(
        "--esm2_num_positions",
        type=int,
        default=16,
        help="Masked residue positions evaluated per protein (default: 16).",
    )
    ap.add_argument(
        "--esm2_max_codons",
        type=int,
        default=128,
        help="Max codons generated per sequence for ESM-2 scoring (default: 128).",
    )
    ap.add_argument(
        "--esm2_seed",
        type=int,
        default=42,
        help="Fixed seed for ESM-2 generation and position sampling (default: 42).",
    )
    ap.add_argument(
        "--esm2_manifest",
        type=Path,
        default=None,
        help="Path to dataset manifest.json for real validation proteins "
             "(default: auto-detected from data/processed/corrected/).",
    )
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    # accept run_id or run_dir
    run_id, run_dir = resolve_run(args.run_id, args.run_dir)

    state_dict, cfg = _load_checkpoint(run_dir)
    model = _build_model_from_cfg(cfg)
    model.load_state_dict(state_dict, strict=False)
    model.to(dev()).eval()

    if args.test_npz:
        test_npz = Path(args.test_npz)
        if not test_npz.is_absolute():
            test_npz = repo / test_npz
    else:
        # test NPZ path from config or combined manifest or default
        test_npz_cfg = cfg.get("test_npz")
        if test_npz_cfg:
            if isinstance(test_npz_cfg, list) and len(test_npz_cfg) > 0:
                test_npz_cfg = test_npz_cfg[0]
            test_npz = Path(test_npz_cfg)
            if not test_npz.is_absolute():
                test_npz = repo / test_npz
        else:
            manifest = repo / "data/processed/combined" / run_id / "manifest.json"
            if manifest.exists():
                test_npz = Path(json.loads(manifest.read_text()).get("test"))
                if not test_npz.is_absolute():
                    test_npz = repo / test_npz
            else:
                test_npz = repo / f"data/processed/test_bs{cfg['block_size']}.npz"

    itos, stoi = load_vocab(run_id, repo)
    kpis = compute_kpis(model, dev(), test_npz, itos, sample_windows=args.max_windows)

    if args.with_esm2:
        print("[sanity_kpis] Running ESM-2 protein critic KPI (this may take ~2 min) ...")
        esm2_kpis = compute_esm2_kpis(
            model,
            dev(),
            run_dir,
            itos,
            stoi,
            num_seqs=args.esm2_num_seqs,
            max_codons=args.esm2_max_codons,
            num_positions=args.esm2_num_positions,
            seed=args.esm2_seed,
            manifest=args.esm2_manifest,
        )
        kpis.update(esm2_kpis)
        delta = esm2_kpis.get("esm2_critic_delta_median", float("nan"))
        p = esm2_kpis.get("esm2_critic_mann_whitney_p")
        status = "PASS" if delta > 0 else "FAIL"
        p_str = f"p={p:.2e}" if p is not None else "p=n/a"
        print(f"[sanity_kpis] ESM-2 KPI: ΔMedian={delta:+.4f} ({p_str}) → {status}")

    metrics_path = repo / "outputs/scores" / run_id / "metrics.json"
    write_merge_metrics(metrics_path, kpis)
    print(json.dumps(kpis, indent=2))


if __name__ == "__main__":
    main()
