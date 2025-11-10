#!/usr/bin/env python3
"""
Sequence quality verifier and realism KPIs for CDS sets.

Computes per-sequence and aggregate metrics:
  - ORF integrity: start codon (ATG/GTG/TTG), no internal stops, terminal stop present
  - Length stats; optional Z-score vs reference length quantiles
  - GC%
  - Codon usage KL/JS divergence vs reference usage (if provided)
  - CAI (Codon Adaptation Index) if reference weights provided or derivable
  - 3-nt periodicity: FFT peak at 1/3 cycles/nt (averaged)
  - Diversity/novelty vs reference set: k-mer Jaccard and MinHash estimate

Inputs:
  --run_id <RUN_ID>             Resolve DNA via runs/<RUN_ID>/pipeline_prepare.json
  --dna path/to/cds.txt         One CDS DNA per line
  Optional references:
    --ref_cds path/to/ref_cds.txt         For length quantiles and diversity baselines
    --ref_usage path/to/codon_usage.tsv   TSV: codon\tfreq for reference
    --ref_cai path/to/cai_weights.tsv     TSV: codon\tw for CAI weights
  Diversity params:
    --kmer 6                           k for Jaccard/MinHash (default 6)
    --minhash 64                       number of hash functions
  Output control:
    --out_dir path                     Where to write summaries/plots (default scores dir for run_id; else outputs/analysis/seq_quality)

Outputs:
  - summary.csv (per-sequence KPIs)
  - plots: periodicity.png (FFT at 1/3), length_hist.png, gc_hist.png
  - Merges headline KPIs into outputs/scores/<RUN_ID>/metrics.json when run_id provided
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


CODONS = [a+b+c for a in "ACGT" for b in "ACGT" for c in "ACGT"]
STOP = {"TAA","TAG","TGA"}
STARTS = {"ATG","GTG","TTG"}


def read_cds(path: Path, limit: int | None = None) -> List[str]:
    seqs = []
    with path.open() as f:
        for line in f:
            s = line.strip().upper().replace("U","T")
            if len(s) >= 9:
                seqs.append(s)
                if limit and len(seqs) >= limit:
                    break
    return seqs


def gc_percent(dna: str) -> float:
    s = dna.upper()
    gc = sum(1 for c in s if c in "GC")
    atgc = sum(1 for c in s if c in "ACGT")
    return 100.0 * gc / max(1, atgc)


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


def kl_div(p: Dict[str, float], q: Dict[str, float]) -> float:
    eps = 1e-12
    return float(sum(p[c] * np.log((p[c]+eps)/(q.get(c,0.0)+eps)) for c in CODONS))


def js_div(p: Dict[str, float], q: Dict[str, float]) -> float:
    m = {c: 0.5*(p.get(c,0.0)+q.get(c,0.0)) for c in CODONS}
    return 0.5*kl_div(p, m) + 0.5*kl_div(q, m)


def load_table_tsv(path: Path) -> Dict[str, float]:
    tbl = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split()  # codon [value]
            if len(parts) >= 2:
                try:
                    tbl[parts[0].upper()] = float(parts[1])
                except ValueError:
                    pass
    return tbl


def derive_cai_weights_from_usage(usage: Dict[str, float]) -> Dict[str, float]:
    # Relative adaptiveness per codon: w_i = f_i / max_{syn}(f) within each AA family
    AA = defaultdict(list)
    # Map codon to AA via simple code (DNA); minimal map for ratio families
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


def cai_for_seq(dna: str, w: Dict[str, float]) -> float:
    L = (len(dna)//3)*3
    vals = []
    for i in range(0, L, 3):
        c = dna[i:i+3]
        if c in STOP:
            break
        if c in w:
            vals.append(max(1e-9, w[c]))
    if not vals:
        return float("nan")
    # CAI = geometric mean of w_i
    return float(np.exp(np.mean(np.log(vals))))


def fft_periodicity_power(dna: str) -> float:
    """Estimate 3-nt periodicity power via FFT on a binary purine indicator as a simple proxy.
    Returns normalized power at f≈1/3 relative to total power.
    """
    if len(dna) < 30:
        return float("nan")
    s = dna.upper()
    x = np.array([1.0 if c in ("A","G") else 0.0 for c in s], dtype=np.float32)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.size, d=1.0)
    # Find closest bin to 1/3 cycles/nt
    idx = int(np.argmin(np.abs(freqs - (1.0/3.0))))
    power = (np.abs(X[idx])**2)
    total = np.sum(np.abs(X)**2) + 1e-12
    return float(power / total)


def kmer_set(dna: str, k: int) -> set:
    s = dna.upper()
    return {s[i:i+k] for i in range(0, len(s)-k+1)} if len(s) >= k else set()


def jaccard_kmer(seqs_a: Iterable[str], seqs_b: Iterable[str], k: int) -> float:
    A = set().union(*[kmer_set(s, k) for s in seqs_a])
    B = set().union(*[kmer_set(s, k) for s in seqs_b])
    if not A and not B:
        return float("nan")
    return float(len(A & B) / max(1, len(A | B)))


def minhash_signature(seqs: Iterable[str], k: int, n_hash: int, seed: int = 1337) -> np.ndarray:
    # Simple MinHash with multiplicative hashes on 64-bit ripemd via Python hash fallback
    import random
    rng = random.Random(seed)
    salts = [rng.randrange(1<<61) for _ in range(n_hash)]
    sig = np.full((n_hash,), np.uint64((1<<64)-1))
    for s in seqs:
        kmers = kmer_set(s, k)
        for hidx, salt in enumerate(salts):
            for kmer in kmers:
                # Compute a simple 64-bit hash; Python's hash is salted, use a stable fallback
                hv = np.uint64(abs(hash(kmer)) ^ salt)  # heuristic; acceptable for relative comparison
                if hv < sig[hidx]:
                    sig[hidx] = hv
    return sig


def minhash_jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    if sig_a.size == 0 or sig_b.size == 0:
        return float("nan")
    return float(np.mean(sig_a == sig_b))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id")
    ap.add_argument("--dna")
    ap.add_argument("--ref_cds")
    ap.add_argument("--ref_usage")
    ap.add_argument("--ref_cai")
    ap.add_argument("--kmer", type=int, default=6)
    ap.add_argument("--minhash", type=int, default=64)
    ap.add_argument("--out_dir")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    if args.run_id and not args.dna:
        run_dir = repo / "runs" / args.run_id
        prep = run_dir / "pipeline_prepare.json"
        if not prep.exists():
            raise SystemExit(f"Cannot resolve DNA: missing {prep}")
        info = json.loads(prep.read_text())
        dna_path = Path(info.get("primary_dna", ""))
        if dna_path and not dna_path.is_absolute():
            dna_path = repo / dna_path
        out_dir = repo / "outputs" / "scores" / args.run_id / "seq_quality"
    else:
        if not args.dna:
            raise SystemExit("Provide --run_id or --dna")
        dna_path = Path(args.dna)
        out_dir = Path(args.out_dir) if args.out_dir else (repo / "outputs" / "analysis" / "seq_quality")
    out_dir.mkdir(parents=True, exist_ok=True)

    seqs = read_cds(dna_path)
    ref_seqs = read_cds(Path(args.ref_cds)) if args.ref_cds else []

    # References
    ref_usage = load_table_tsv(Path(args.ref_usage)) if args.ref_usage else None
    if ref_usage is None and ref_seqs:
        ref_usage = codon_usage(ref_seqs)
    ref_cai_w = load_table_tsv(Path(args.ref_cai)) if args.ref_cai else None
    if ref_cai_w is None and ref_usage is not None:
        ref_cai_w = derive_cai_weights_from_usage(ref_usage)

    # Per-sequence metrics
    rows = []
    periodicities = []
    lengths = []
    gcs = []
    for i, dna in enumerate(seqs):
        L = (len(dna)//3)*3
        starts_ok = dna[:3] in STARTS
        # internal stop check
        internal_stop = False
        for j in range(3, L-3, 3):
            if dna[j:j+3] in STOP:
                internal_stop = True; break
        terminal_stop = dna[L-3:L] in STOP if L >= 3 else False
        orf_ok = starts_ok and (not internal_stop) and terminal_stop
        gc = gc_percent(dna)
        cai = float("nan")
        if ref_cai_w:
            try:
                cai = cai_for_seq(dna, ref_cai_w)
            except Exception:
                cai = float("nan")
        pow3 = fft_periodicity_power(dna)
        lengths.append(L//3)
        gcs.append(gc)
        if not np.isnan(pow3):
            periodicities.append(pow3)
        rows.append({
            "id": i,
            "len_codon": L//3,
            "orf_ok": int(orf_ok),
            "start_ok": int(starts_ok),
            "terminal_stop": int(terminal_stop),
            "internal_stop": int(internal_stop),
            "gc_percent": gc,
            "cai": cai,
            "fft_1over3": pow3,
        })

    # Codon usage divergences
    usage = codon_usage(seqs)
    usage_kl = usage_js = float("nan")
    if ref_usage is not None:
        usage_kl = kl_div(usage, ref_usage)
        usage_js = js_div(usage, ref_usage)

    # Diversity vs ref
    jacc = minhash_j = float("nan")
    if ref_seqs:
        jacc = jaccard_kmer(seqs, ref_seqs, k=args.kmer)
        sig_a = minhash_signature(seqs, k=args.kmer, n_hash=args.minhash)
        sig_b = minhash_signature(ref_seqs, k=args.kmer, n_hash=args.minhash)
        minhash_j = minhash_jaccard(sig_a, sig_b)

    # Write summary CSV
    summary_csv = out_dir / "summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["id","len_codon"]) 
        writer.writeheader(); [writer.writerow(r) for r in rows]

    # Plots
    if lengths:
        plt.figure(figsize=(5,3)); plt.hist(lengths, bins=30, alpha=0.85)
        plt.xlabel('length (codons)'); plt.ylabel('count'); plt.tight_layout(); plt.savefig(out_dir/"length_hist.png"); plt.close()
    if gcs:
        plt.figure(figsize=(5,3)); plt.hist(gcs, bins=30, alpha=0.85)
        plt.xlabel('GC%'); plt.ylabel('count'); plt.tight_layout(); plt.savefig(out_dir/"gc_hist.png"); plt.close()
    if periodicities:
        plt.figure(figsize=(5,3)); plt.hist(periodicities, bins=30, alpha=0.85)
        plt.xlabel('FFT power at 1/3'); plt.ylabel('count'); plt.tight_layout(); plt.savefig(out_dir/"periodicity.png"); plt.close()

    # Merge KPIs into metrics.json (run_id only)
    if args.run_id:
        import statistics as stats
        metrics_path = repo / "outputs" / "scores" / args.run_id / "metrics.json"
        try:
            metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        except Exception:
            metrics = {}
        metrics.update({
            "orf_ok_rate": float(np.mean([r["orf_ok"] for r in rows])) if rows else float("nan"),
            "len_codon_median": float(np.median(lengths)) if lengths else float("nan"),
            "gc_percent_median": float(np.median(gcs)) if gcs else float("nan"),
            "fft_1over3_median": float(np.median(periodicities)) if periodicities else float("nan"),
            "codon_usage_kl": usage_kl,
            "codon_usage_js": usage_js,
            "kmer_jaccard": jacc,
            "minhash_jaccard": minhash_j,
        })
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
        print(f"[seq-quality] merged KPIs into {metrics_path}")

    print(f"[seq-quality] wrote {summary_csv} and plots under {out_dir}")


if __name__ == "__main__":
    main()

