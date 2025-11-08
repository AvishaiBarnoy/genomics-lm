#!/usr/bin/env python3
"""
Compute sanity KPIs on a held-out subset of test windows and merge them into metrics.json.

KPIs:
- codon_corr: Pearson correlation between model unigram codon distribution (aggregated next-token probs) and corpus codon frequencies.
- frameshift_delta: mean ΔNLL when shifting the input window by +1 token (codon).
- startstop_delta.start / .stop: mean ΔNLL when mutating a start (ATG) / stop (TAA/TAG/TGA) token in-context.
- syn_gap: average (ΔNLL_nonSyn − ΔNLL_syn).

Keeps runtime small by sampling at most N windows.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.codonlm.model_tiny_gpt import TinyGPT
from src.codonlm.metrics_io import write_merge_metrics
from scripts._shared import resolve_run


def dev() -> torch.device:
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


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


def _load_checkpoint(run_dir: Path) -> tuple[dict, dict]:
    best = run_dir / "best.pt"
    state = torch.load(best, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        return state["model"], state.get("cfg", {})
    return state, {}


def _build_model_from_cfg(cfg: dict) -> TinyGPT:
    return TinyGPT(
        vocab_size=int(cfg["vocab_size"]),
        block_size=int(cfg["block_size"]),
        n_layer=int(cfg["n_layer"]),
        n_head=int(cfg["n_head"]),
        n_embd=int(cfg["n_embd"]),
        dropout=float(cfg.get("dropout", 0.0)),
        use_checkpoint=False,
        label_smoothing=float(cfg.get("label_smoothing", 0.0)),
    )


class NPZ(Dataset):
    def __init__(self, path: Path):
        blob = np.load(path)
        self.X = torch.from_numpy(np.asarray(blob["X"]).astype(np.int64))
        self.Y = torch.from_numpy(np.asarray(blob["Y"]).astype(np.int64))
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, i):
        return self.X[i], self.Y[i]


def load_vocab(run_id: str, repo_root: Path) -> Tuple[List[str], Dict[str, int]]:
    itos_path = repo_root / "runs" / run_id / "itos.txt"
    tokens = [line.strip() for line in itos_path.read_text().splitlines() if line.strip()]
    stoi = {tok: i for i, tok in enumerate(tokens)}
    return tokens, stoi


@torch.no_grad()
def batch_nll(model: TinyGPT, device: torch.device, xb: torch.Tensor, yb: torch.Tensor) -> float:
    xb = xb.to(device)
    yb = yb.to(device)
    logits, loss = model(xb, yb)
    if loss is None:
        return 0.0
    valid = (yb != 0).sum().item()
    return float(loss.item()) * max(1, valid)


def codon_mask(itos: List[str]) -> np.ndarray:
    return np.array([1 if (len(tok) == 3 and all(c in "ACGT" for c in tok)) else 0 for tok in itos], dtype=np.int32)


def compute_kpis(model: TinyGPT, device: torch.device, test_npz: Path, itos: List[str], sample_windows: int = 200) -> Dict[str, float]:
    ds = NPZ(test_npz)
    n = min(sample_windows, len(ds))
    idxs = np.linspace(0, len(ds) - 1, num=n, dtype=int)
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
            return float('nan')
        nll_m = batch_nll(model, device, xb_mut, y)
        return float((nll_m - nll_orig) / max(1, (y != 0).sum().item()))

    start_delta = mutate_once(x.clone(), start_idx, non_start) if start_idx >= 0 else float("nan")
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
            syn_ids = [j for j, t in enumerate(itos) if t in CODON_TO_AA and CODON_TO_AA[t] == aa and t != tok_str]
            nonsyn_ids = [j for j, t in enumerate(itos) if t in CODON_TO_AA and CODON_TO_AA[t] != aa and t not in ("TAA","TAG","TGA")]
            if not syn_ids or not nonsyn_ids:
                continue
            # mutate and measure per-position delta
            row_syn = row.clone(); row_syn[p] = int(random.choice(syn_ids))
            row_nsn = row.clone(); row_nsn[p] = int(random.choice(nonsyn_ids))
            # batch both with same y context length adjust
            xb = torch.stack([row, row_syn, row_nsn], dim=0)
            nlls = batch_nll(model, device, xb, y[i:i+1].repeat(3, 1))  # returns sum over tokens; but our helper expects batch; adjust helper? reuse as is
            # Since batch_nll returns sum over valid tokens for entire batch, get average per example by dividing by 3 and token count is same across
            # To keep simple, compute deltas via separate calls
            nll0 = batch_nll(model, device, row.unsqueeze(0), y[i:i+1])
            nllS = batch_nll(model, device, row_syn.unsqueeze(0), y[i:i+1])
            nllN = batch_nll(model, device, row_nsn.unsqueeze(0), y[i:i+1])
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", help="outputs/checkpoints/<RUN_ID>")
    ap.add_argument("--run_id", help="Run id (alternative to --run_dir)")
    ap.add_argument("--max_windows", type=int, default=200)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    # accept run_id or run_dir
    run_id, _ = resolve_run(args.run_id, args.run_dir)
    run_dir = Path(args.run_dir) if args.run_dir else (repo / "outputs" / "checkpoints" / run_id)

    state_dict, cfg = _load_checkpoint(run_dir)
    model = _build_model_from_cfg(cfg)
    model.load_state_dict(state_dict, strict=False)
    model.to(dev()).eval()

    # test NPZ path from combined manifest or default
    manifest = repo / "data/processed/combined" / run_id / "manifest.json"
    if manifest.exists():
        test_npz = Path(json.loads(manifest.read_text()).get("test"))
        if not test_npz.is_absolute():
            test_npz = repo / test_npz
    else:
        test_npz = repo / f"data/processed/test_bs{cfg['block_size']}.npz"

    itos, _ = load_vocab(run_id, repo)
    kpis = compute_kpis(model, dev(), test_npz, itos, sample_windows=args.max_windows)
    metrics_path = repo / "outputs/scores" / run_id / "metrics.json"
    write_merge_metrics(metrics_path, kpis)
    print(json.dumps(kpis, indent=2))


if __name__ == "__main__":
    main()
