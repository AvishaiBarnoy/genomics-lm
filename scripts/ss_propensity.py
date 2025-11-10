#!/usr/bin/env python3
"""
Heuristic secondary-structure propensity analysis for CDS-derived proteins.

Computes α-helix (H) and β-sheet (E) propensities from amino-acid sequences
derived from DNA/CDS, using simple sliding-window rules (Chou–Fasman style).

Inputs (choose one):
  --dna path/to/cds.txt     (one DNA CDS per line, ACGT)
  --run_id <RUN_ID>         (resolves runs/<RUN_ID>/pipeline_prepare.json → primary_dna)

Outputs:
  - summary.csv with per-sequence counts and fractions
  - segment length histograms (PNG)
  - if run_id provided: merges a few summary KPIs into outputs/scores/<RUN_ID>/metrics.json

Note: This is heuristic/correlation-level — not a structure predictor. Use as
      a sanity/realism metric to complement LM benchmarks.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt


# Codon → AA table (DNA)
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

# Heuristic propensities (normalized ~1 baselines)
# These are approximate and chosen for relative ranking, not absolute prediction.
HELIX_PROP = {
    "A":1.42,"R":1.21,"N":0.67,"D":1.01,"C":0.70,"Q":1.11,"E":1.51,"G":0.57,"H":1.00,"I":1.08,
    "L":1.21,"K":1.16,"M":1.45,"F":1.13,"P":0.57,"S":0.77,"T":0.83,"W":1.08,"Y":0.69,"V":1.06,
}
SHEET_PROP = {
    "A":0.83,"R":0.72,"N":0.89,"D":0.54,"C":1.19,"Q":1.10,"E":0.37,"G":0.75,"H":0.87,"I":1.60,
    "L":1.30,"K":0.74,"M":1.05,"F":1.38,"P":0.55,"S":0.75,"T":1.19,"W":1.37,"Y":1.47,"V":1.70,
}


def dna_to_aa(dna: str) -> List[str]:
    s = dna.strip().upper().replace("U","T")
    L = (len(s)//3)*3
    aas: List[str] = []
    for i in range(0, L, 3):
        aa = CODON_TO_AA.get(s[i:i+3], "?")
        if aa == "*":  # stop
            break
        if aa != "?":
            aas.append(aa)
    return aas


def segments_from_propensity(aas: List[str], table: Dict[str, float], win: int, thr: float) -> List[Tuple[int,int]]:
    segs: List[Tuple[int,int]] = []
    i = 0
    while i + win <= len(aas):
        window = aas[i:i+win]
        vals = [table.get(a, 1.0) for a in window]
        if sum(vals)/win >= thr:
            # extend while mean stays above thr
            j = i + win
            while j < len(aas) and table.get(aas[j], 1.0) >= thr:
                j += 1
            segs.append((i, j))
            i = j
        else:
            i += 1
    return segs


def summarize_segments(segs: List[Tuple[int,int]], L: int) -> Dict[str, float]:
    if not segs:
        return {"count":0, "max_len":0, "frac":0.0}
    lengths = [b-a for a,b in segs]
    return {
        "count": len(segs),
        "max_len": max(lengths),
        "frac": float(sum(lengths))/max(1, L)
    }


def merge_metrics(metrics_path: Path, updates: Dict[str, float]) -> None:
    metrics = {}
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text())
        except Exception:
            metrics = {}
    metrics.update({k: float(v) for k,v in updates.items()})
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dna", help="Path to CDS DNA file (one per line)")
    ap.add_argument("--run_id", help="Resolve DNA path from run’s pipeline_prepare.json")
    ap.add_argument("--out_dir", help="Where to write outputs")
    ap.add_argument("--helix_win", type=int, default=12)
    ap.add_argument("--sheet_win", type=int, default=4)
    ap.add_argument("--helix_thr", type=float, default=1.1)
    ap.add_argument("--sheet_thr", type=float, default=1.1)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir) if args.out_dir else (repo / "outputs" / "analysis")

    if args.run_id and not args.dna:
        run_dir = repo / "runs" / args.run_id
        prep = run_dir / "pipeline_prepare.json"
        if not prep.exists():
            raise SystemExit(f"Cannot resolve DNA: missing {prep}")
        data = json.loads(prep.read_text())
        dna_path = Path(data["primary_dna"]) if data.get("primary_dna") else None
        if dna_path and not dna_path.is_absolute():
            dna_path = repo / dna_path
        if not (dna_path and dna_path.exists()):
            raise SystemExit("Resolved DNA path not found")
        dna_file = dna_path
        out_dir = repo / "outputs" / "scores" / args.run_id / "ss_propensity"
    else:
        if not args.dna:
            raise SystemExit("Provide --dna or --run_id")
        dna_file = Path(args.dna)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Read DNA
    seqs: List[str] = []
    with dna_file.open() as f:
        for line in f:
            s = line.strip().upper().replace("U","T")
            if len(s) >= 9:
                seqs.append(s)

    rows = []
    helix_lengths: List[int] = []
    sheet_lengths: List[int] = []
    for i, dna in enumerate(seqs):
        aa = dna_to_aa(dna)
        hsegs = segments_from_propensity(aa, HELIX_PROP, args.helix_win, args.helix_thr)
        esegs = segments_from_propensity(aa, SHEET_PROP, args.sheet_win, args.sheet_thr)
        hsum = summarize_segments(hsegs, len(aa))
        esum = summarize_segments(esegs, len(aa))
        rows.append({
            "id": i,
            "len_aa": len(aa),
            "helix_segments": hsum["count"],
            "sheet_segments": esum["count"],
            "max_helix_len": hsum["max_len"],
            "max_sheet_len": esum["max_len"],
            "helix_frac": hsum["frac"],
            "sheet_frac": esum["frac"],
        })
        helix_lengths += [b-a for a,b in hsegs]
        sheet_lengths += [b-a for a,b in esegs]

    # Write summary CSV
    summary_csv = out_dir / "summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["id","len_aa"])
        writer.writeheader(); [writer.writerow(r) for r in rows]

    # Plots
    for name, lengths in [("helix", helix_lengths), ("sheet", sheet_lengths)]:
        if not lengths:
            continue
        plt.figure(figsize=(5,3))
        plt.hist(lengths, bins=range(0, max(lengths)+2, 2), alpha=0.8)
        plt.xlabel(f"{name} segment length (aa)"); plt.ylabel("count"); plt.title(f"{name} segment length distribution")
        plt.tight_layout(); plt.savefig(out_dir / f"{name}_length_hist.png"); plt.close()

    # Merge a few KPIs into metrics.json if run_id mode
    if args.run_id:
        import statistics as stats
        metrics_path = repo / "outputs" / "scores" / args.run_id / "metrics.json"
        updates = {}
        if helix_lengths:
            updates["ss_helix_medlen"] = float(stats.median(helix_lengths))
        if sheet_lengths:
            updates["ss_sheet_medlen"] = float(stats.median(sheet_lengths))
        merge_metrics(metrics_path, updates)
        print(f"[ss] merged KPIs into {metrics_path}")

    print(f"[ss] wrote {summary_csv} and plots under {out_dir}")


if __name__ == "__main__":
    main()

