#!/usr/bin/env python3
"""
Disorder heuristics for CDS-derived protein sequences.

Computes simple, literature-inspired indicators of intrinsic disorder:
  - Charge–hydropathy (Uversky) point + rule-of-thumb classification
  - Net charge per residue (NCPR) and a simple charge-patterning proxy (kappa-like)
  - Fraction of disorder-promoting residues (E,D,K,R,Q,S,P,G)
  - Low-complexity segments via Shannon entropy windows (SEG-like heuristic)

Inputs (choose one):
  --run_id <RUN_ID>   resolves DNA path via runs/<RUN_ID>/pipeline_prepare.json
  --dna path/to/cds.txt (one CDS per line)

Outputs:
  - summary.csv with per-sequence KPIs
  - Plots: ch_plane.png, lcr_len_hist.png, disorder_frac_hist.png
  - If --run_id given: merges aggregate KPIs into outputs/scores/<RUN_ID>/metrics.json

Note: Heuristics, not predictors. Use to complement secondary-structure (SS) analysis.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


# DNA codon → AA (stop='*')
CODON_TO_AA: Dict[str, str] = {
    "TTT":"F","TTC":"F","TTA":"L","TTG":"L","TCT":"S","TCC":"S","TCA":"S","TCG":"S",
    "TAT":"Y","TAC":"Y","TAA":"*","TAG":"*","TGT":"C","TGC":"C","TGA":"*","TGG":"W",
    "CTT":"L","CTC":"L","CTA":"L","CTG":"L","CCT":"P","CCC":"P","CCA":"P","CCG":"P",
    "CAT":"H","CAC":"H","CAA":"Q","CAG":"Q","CGT":"R","CGC":"R","CGA":"R","CGG":"R",
    "ATT":"I","ATC":"I","ATA":"I","ATG":"M","ACT":"T","ACC":"T","ACA":"T","ACG":"T",
    "AAT":"N","AAC":"N","AAA":"K","AAG":"K","AGT":"S","AGC":"S","AGA":"R","AGG":"R",
    "GTT":"V","GTC":"V","GTA":"V","GTG":"V","GCT":"A","GCC":"A","GCA":"A","GCG":"A",
    "GAT":"D","GAC":"D","GAA":"E","GAG":"E","GGT":"G","GGC":"G","GGA":"G","GGG":"G",
}

# Kyte–Doolittle hydropathy scale (approx)
KD = {
    "I":4.5,"V":4.2,"L":3.8,"F":2.8,"C":2.5,"M":1.9,"A":1.8,"G":-0.4,"T":-0.7,
    "S":-0.8,"W":-0.9,"Y":-1.3,"P":-1.6,"H":-3.2,"E":-3.5,"Q":-3.5,"D":-3.5,"N":-3.5,"K":-3.9,"R":-4.5
}

# Residue charges at ~pH 7 (coarse)
CHARGE = {"K":+1,"R":+1,"H":+0.1,"D":-1,"E":-1}

DISORDER_SET = set(list("EDKRQSPG"))


def dna_to_aa(dna: str) -> List[str]:
    s = dna.strip().upper().replace("U","T")
    L = (len(s)//3)*3
    aas: List[str] = []
    for i in range(0, L, 3):
        aa = CODON_TO_AA.get(s[i:i+3], "?")
        if aa == "*":
            break
        if aa != "?":
            aas.append(aa)
    return aas


def mean_hydropathy(aas: List[str]) -> float:
    vals = [KD.get(a, 0.0) for a in aas]
    return float(np.mean(vals)) if vals else 0.0


def ncpr(aas: List[str]) -> float:
    ch = [CHARGE.get(a, 0.0) for a in aas]
    return float(np.sum(ch)) / max(1, len(aas))


def kappa_proxy(aas: List[str]) -> float:
    """Simple charge patterning proxy: mean absolute product of neighboring charges.
    Higher when like charges cluster; zero when uncharged or alternating.
    """
    if len(aas) < 2:
        return 0.0
    ch = np.array([CHARGE.get(a, 0.0) for a in aas], dtype=float)
    return float(np.mean(np.abs(ch[:-1] * ch[1:])))


def disorder_fraction(aas: List[str]) -> float:
    if not aas:
        return 0.0
    return float(sum(1 for a in aas if a in DISORDER_SET)) / len(aas)


def low_complexity_segments(aas: List[str], win: int = 12, thr_entropy: float = 1.8) -> List[Tuple[int,int]]:
    """SEG-like heuristic using Shannon entropy threshold (lower = simpler).
    thr_entropy ~1.8 is a common ballpark for low complexity windows.
    """
    segs: List[Tuple[int,int]] = []
    i = 0
    while i + win <= len(aas):
        window = aas[i:i+win]
        # entropy in bits
        uniq, counts = np.unique(window, return_counts=True)
        p = counts / counts.sum()
        ent = -np.sum(p * np.log2(p))
        if ent <= thr_entropy:
            j = i + win
            while j < len(aas):
                w2 = aas[j-win+1:j+1]
                uniq2, counts2 = np.unique(w2, return_counts=True)
                p2 = counts2 / counts2.sum()
                ent2 = -np.sum(p2 * np.log2(p2))
                if ent2 <= thr_entropy:
                    j += 1
                else:
                    break
            segs.append((i, j))
            i = j
        else:
            i += 1
    return segs


def classify_uversky(mean_kd: float, net_charge: float) -> str:
    # Uversky boundary (approx for IDP classification): R = 2.785*H - 1.151
    # Treat above the line as disordered (higher charge at given hydropathy)
    boundary = 2.785 * mean_kd - 1.151
    return "disordered" if net_charge > boundary else "folded_like"


def merge_metrics(path: Path, updates: Dict[str, float]) -> None:
    metrics = {}
    if path.exists():
        try:
            metrics = json.loads(path.read_text())
        except Exception:
            metrics = {}
    metrics.update({k: float(v) for k, v in updates.items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id")
    ap.add_argument("--dna")
    ap.add_argument("--out_dir")
    ap.add_argument("--lcr_win", type=int, default=12)
    ap.add_argument("--lcr_entropy", type=float, default=1.8)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    if args.run_id and not args.dna:
        run_dir = repo / "runs" / args.run_id
        prep = run_dir / "pipeline_prepare.json"
        if not prep.exists():
            raise SystemExit(f"Cannot resolve DNA: missing {prep}")
        info = json.loads(prep.read_text())
        dna_path = Path(info["primary_dna"]) if info.get("primary_dna") else None
        if dna_path and not dna_path.is_absolute():
            dna_path = repo / dna_path
        if not (dna_path and dna_path.exists()):
            raise SystemExit("Resolved DNA path not found")
        dna_file = dna_path
        out_dir = repo / "outputs" / "scores" / args.run_id / "disorder"
    else:
        if not args.dna:
            raise SystemExit("Provide --run_id or --dna")
        dna_file = Path(args.dna)
        out_dir = Path(args.out_dir) if args.out_dir else (repo / "outputs" / "analysis" / "disorder")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Read DNA sequences
    seqs: List[str] = []
    with dna_file.open() as f:
        for line in f:
            s = line.strip().upper().replace("U","T")
            if len(s) >= 9:
                seqs.append(s)

    rows = []
    lcr_lengths: List[int] = []
    disorder_fracs: List[float] = []
    ch_pts: List[Tuple[float,float]] = []
    for i, dna in enumerate(seqs):
        aa = dna_to_aa(dna)
        if not aa:
            continue
        H = mean_hydropathy(aa)
        R = ncpr(aa)
        cls = classify_uversky(H, R)
        frac_dis = disorder_fraction(aa)
        segs = low_complexity_segments(aa, win=args.lcr_win, thr_entropy=args.lcr_entropy)
        lcr_len = [b-a for a,b in segs]
        kappa = kappa_proxy(aa)
        rows.append({
            "id": i,
            "len_aa": len(aa),
            "mean_hydro": H,
            "ncpr": R,
            "uversky": cls,
            "disorder_frac": frac_dis,
            "lcr_frac": float(sum(lcr_len))/max(1, len(aa)),
            "max_lcr_len": (max(lcr_len) if lcr_len else 0),
            "kappa_proxy": kappa,
        })
        ch_pts.append((H, R))
        lcr_lengths += lcr_len
        disorder_fracs.append(frac_dis)

    # Write summary
    summary_csv = out_dir / "summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["id","len_aa"])
        writer.writeheader(); [writer.writerow(r) for r in rows]

    # Plots
    if ch_pts:
        xs, ys = zip(*ch_pts)
        plt.figure(figsize=(5,4))
        plt.scatter(xs, ys, s=10, alpha=0.6)
        # Uversky boundary line over a reasonable H range
        Hs = np.linspace(min(xs)-0.5, max(xs)+0.5, 100)
        boundary = 2.785*Hs - 1.151
        plt.plot(Hs, boundary, 'r--', label='Uversky boundary')
        plt.xlabel('mean hydropathy (KD)'); plt.ylabel('net charge per residue')
        plt.legend(); plt.tight_layout(); plt.savefig(out_dir/"ch_plane.png"); plt.close()
    if lcr_lengths:
        plt.figure(figsize=(5,3))
        bins = range(0, max(lcr_lengths)+2, 2)
        plt.hist(lcr_lengths, bins=bins, alpha=0.85)
        plt.xlabel('low-complexity segment length (aa)'); plt.ylabel('count')
        plt.tight_layout(); plt.savefig(out_dir/"lcr_len_hist.png"); plt.close()
    if disorder_fracs:
        plt.figure(figsize=(5,3))
        plt.hist(disorder_fracs, bins=20, alpha=0.85)
        plt.xlabel('disorder-promoting residue fraction'); plt.ylabel('count')
        plt.tight_layout(); plt.savefig(out_dir/"disorder_frac_hist.png"); plt.close()

    # Merge some KPIs into metrics.json when run_id used
    if args.run_id:
        import statistics as stats
        metrics_path = repo / "outputs" / "scores" / args.run_id / "metrics.json"
        agg = {}
        if rows:
            agg["disorder_frac_median"] = float(stats.median(disorder_fracs)) if disorder_fracs else 0.0
            agg["max_lcr_len_median"] = float(stats.median(lcr_lengths)) if lcr_lengths else 0.0
        merge_metrics(metrics_path, agg)
        print(f"[disorder] merged KPIs into {metrics_path}")

    print(f"[disorder] wrote {summary_csv} and plots under {out_dir}")


if __name__ == "__main__":
    main()

