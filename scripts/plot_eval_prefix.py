#!/usr/bin/env python3
"""Plot prefix-generation summary with combined axes.

Reads summary.csv produced by scripts/eval_generation_prefix.py and writes a dual-axis plot:
  left y-axis: median_gqs (and median_gqs_norm if present)
  right y-axis: mean_aa_identity

CLI:
  python -m scripts.plot_eval_prefix --summary outputs/scores/<RUN_ID>/gen_prefix/summary.csv --out_dir outputs/figs
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt


def load_summary(path: Path) -> List[Dict[str, float]]:
    rows = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append({k: (float(v) if k != "k" and v != "" else int(row["k"])) for k, v in row.items()})
            except Exception:
                # Cast manually
                r2 = {}
                for k, v in row.items():
                    if k == "k":
                        r2[k] = int(v)
                    else:
                        try:
                            r2[k] = float(v)
                        except Exception:
                            pass
                rows.append(r2)
    rows.sort(key=lambda r: r.get("k", 0))
    return rows


def plot_dual_axis(rows: List[Dict[str, float]], out_path: Path) -> None:
    ks = [r["k"] for r in rows]
    med_gqs = [r.get("median_gqs") for r in rows]
    med_gqs_norm = [r.get("median_gqs_norm") for r in rows] if any("median_gqs_norm" in r for r in rows) else None
    mean_aa = [r.get("mean_aa_identity") for r in rows]

    fig, ax1 = plt.subplots(figsize=(6,4))
    ax1.set_xlabel("k (prefix codons)")
    ax1.set_ylabel("median_gqs", color="tab:blue")
    ax1.plot(ks, med_gqs, marker="o", color="tab:blue", label="median_gqs")
    if med_gqs_norm is not None and any(x is not None for x in med_gqs_norm):
        ax1.plot(ks, med_gqs_norm, marker="x", color="tab:cyan", label="median_gqs_norm")
    ax1.tick_params(axis='y', labelcolor="tab:blue")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.set_ylabel("mean_aa_identity", color="tab:orange")
    ax2.plot(ks, mean_aa, marker="s", color="tab:orange", label="mean_aa_identity")
    ax2.tick_params(axis='y', labelcolor="tab:orange")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    rows = load_summary(Path(args.summary))
    out = Path(args.out_dir) / "gqs_identity_vs_k.png"
    plot_dual_axis(rows, out)
    print(f"[plot] wrote {out}")


if __name__ == "__main__":
    main()

