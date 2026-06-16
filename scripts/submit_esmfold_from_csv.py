#!/usr/bin/env python3
"""Submit generated protein sequences from a design CSV to ESMFold.

This is a resume helper for cases where generation succeeded but network access
failed during the ESMFold phase.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from scripts.generative_design_loop import esm_fold


def _as_float(value: str | None) -> float:
    try:
        return float(value or 0.0)
    except ValueError:
        return 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Submit AA sequences in a design CSV to ESMFold.")
    ap.add_argument("--csv", required=True, help="Input CSV with aa_seq column")
    ap.add_argument("--out_dir", required=True, help="Directory for PDBs and esm_fold_results.csv")
    ap.add_argument("--top", type=int, default=0, help="Submit top N by stability_prob; 0 means all")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(csv_path.open()))
    rows = [r for r in rows if r.get("aa_seq")]
    rows.sort(key=lambda r: _as_float(r.get("stability_prob")), reverse=True)
    if args.top > 0:
        rows = rows[:args.top]

    results: list[dict] = []
    for rank, row in enumerate(rows, 1):
        seq_id = row.get("seq_id", str(rank))
        print(
            f"[esm] {rank}/{len(rows)} seq_id={seq_id} "
            f"stability={_as_float(row.get('stability_prob')):.3f} "
            f"len={len(row.get('aa_seq', ''))}"
        )
        fold = esm_fold(row["aa_seq"], timeout=120)
        result = {
            "rank": rank,
            "seq_id": seq_id,
            "prefix_id": row.get("prefix_id", ""),
            "stability_prob": row.get("stability_prob", ""),
            "n_aa": row.get("n_aa", ""),
        }
        if fold:
            pdb_path = out_dir / f"top_{rank}_seq{seq_id}.pdb"
            pdb_path.write_text(fold["pdb_text"])
            result.update({
                "plddt_mean": fold["plddt_mean"],
                "plddt_min": fold["plddt_min"],
                "plddt_max": fold["plddt_max"],
                "pdb_path": str(pdb_path),
                "status": "ok",
            })
            print(f"      pLDDT={fold['plddt_mean']:.3f} saved={pdb_path.name}")
        else:
            result["status"] = "failed"
            print("      failed")
        results.append(result)

    out_csv = out_dir / "esm_fold_results.csv"
    fieldnames = [
        "rank",
        "seq_id",
        "prefix_id",
        "stability_prob",
        "n_aa",
        "plddt_mean",
        "plddt_min",
        "plddt_max",
        "pdb_path",
        "status",
    ]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"[esm] wrote {out_csv}")


if __name__ == "__main__":
    main()
