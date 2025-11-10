#!/usr/bin/env python3
"""
Build organism reference tables from CDS for downstream quality metrics.

Given a CDS file (one DNA sequence per line) or --run_id to resolve primary_dna,
computes:
  - codon_usage.tsv  (codon\tfreq)
  - cai_weights.tsv  (codon\tw)   relative adaptiveness per AA family

Outputs to: data/reference/<name>/

CLI:
  python -m scripts.build_reference_tables --name ecoli_k12 --cds data/processed/ecoli_k12/cds_dna.txt
  python -m scripts.build_reference_tables --name ecoli_k12 --run_id <RUN_ID>

Notes: If you maintain multiple datasets, run once per organism name.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Dict, Iterable, List


CODONS = [a+b+c for a in "ACGT" for b in "ACGT" for c in "ACGT"]


def read_cds(path: Path) -> List[str]:
    seqs = []
    with path.open() as f:
        for line in f:
            s = line.strip().upper().replace("U","T")
            if len(s) >= 9:
                seqs.append(s)
    return seqs


def codon_usage(seqs: Iterable[str]) -> Dict[str, float]:
    cnt = Counter()
    total = 0
    for dna in seqs:
        L = (len(dna)//3)*3
        for i in range(0, L, 3):
            cod = dna[i:i+3]
            if len(cod) == 3:
                cnt[cod] += 1
                total += 1
    if total == 0:
        return {c: 0.0 for c in CODONS}
    return {c: cnt.get(c, 0) / total for c in CODONS}


def derive_cai_weights_from_usage(usage: Dict[str, float]) -> Dict[str, float]:
    genetic = {
        "TTT":"F","TTC":"F","TTA":"L","TTG":"L","TCT":"S","TCC":"S","TCA":"S","TCG":"S",
        "TAT":"Y","TAC":"Y","TAA":"*","TAG":"*","TGT":"C","TGC":"C","TGA":"*","TGG":"W",
        "CTT":"L","CTC":"L","CTA":"L","CTG":"L","CCT":"P","CCC":"P","CCA":"P","CCG":"P",
        "CAT":"H","CAC":"H","CAA":"Q","CAG":"Q","CGT":"R","CGC":"R","CGA":"R","CGG":"R",
        "ATT":"I","ATC":"I","ATA":"I","ATG":"M","ACT":"T","ACC":"T","ACA":"T","ACG":"T",
        "AAT":"N","AAC":"N","AAA":"K","AAG":"K","AGT":"S","AGC":"S","AGA":"R","AGG":"R",
        "GTT":"V","GTC":"V","GTA":"V","GTG":"V","GCT":"A","GCC":"A","GCA":"A","GCG":"A",
        "GAT":"D","GAC":"D","GAA":"E","GAG":"E","GGT":"G","GGC":"G","GGA":"G","GGG":"G",
    }
    AA = defaultdict(list)
    for cod, f in usage.items():
        aa = genetic.get(cod)
        if aa and aa != "*":
            AA[aa].append((cod, f))
    weights = {}
    for aa, items in AA.items():
        maxf = max((f for _, f in items), default=0.0)
        if maxf <= 0:
            for cod, _ in items:
                weights[cod] = 0.0
        else:
            for cod, f in items:
                weights[cod] = f / maxf
    return weights


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="Organism/dataset name (folder under data/reference)")
    ap.add_argument("--cds", help="Path to CDS DNA file (one per line)")
    ap.add_argument("--run_id", help="Resolve CDS from runs/<RUN_ID>/pipeline_prepare.json")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    if args.run_id and not args.cds:
        prep = repo / "runs" / args.run_id / "pipeline_prepare.json"
        if not prep.exists():
            raise SystemExit(f"Cannot resolve CDS: missing {prep}")
        info = json.loads(prep.read_text())
        cds_path = Path(info.get("primary_dna", ""))
        if cds_path and not cds_path.is_absolute():
            cds_path = repo / cds_path
    else:
        if not args.cds:
            raise SystemExit("Provide --cds or --run_id")
        cds_path = Path(args.cds)
    if not cds_path.exists():
        raise SystemExit(f"CDS file not found: {cds_path}")

    seqs = read_cds(cds_path)
    usage = codon_usage(seqs)
    weights = derive_cai_weights_from_usage(usage)

    out_dir = repo / "data" / "reference" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "codon_usage.tsv").write_text("\n".join(f"{c}\t{usage.get(c,0.0):.8f}" for c in CODONS) + "\n")
    (out_dir / "cai_weights.tsv").write_text("\n".join(f"{c}\t{weights.get(c,0.0):.8f}" for c in CODONS) + "\n")
    print(f"[build-ref] wrote {out_dir}/codon_usage.tsv and {out_dir}/cai_weights.tsv from {cds_path}")


if __name__ == "__main__":
    import json
    main()
