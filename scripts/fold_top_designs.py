#!/usr/bin/env python3
"""
Fold the top sequence from each ablation configuration using ESMFold and print physical pLDDT values.
"""

import csv
import json
import requests
from pathlib import Path
import numpy as np

def esm_fold(aa_seq: str, timeout: int = 45) -> dict:
    url = "https://api.esmatlas.com/foldSequence/v1/pdb/"
    try:
        resp = requests.post(url, data=aa_seq, timeout=timeout,
                             headers={"Content-Type": "application/x-www-form-urlencoded"})
        if resp.status_code != 200:
            print(f"  [ESMFold] API returned status code {resp.status_code}")
            return None
        pdb_text = resp.text
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
        print(f"  [ESMFold] API error: {exc}")
        return None

def get_top_sequence(csv_path: Path) -> tuple[dict | None, str | None]:
    if not csv_path.exists():
        return None, None
    records = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    if not records:
        return None, None
    stability_key = next(
        (key for key in (
            "stability_megascale_delta_g_pred_kcal_mol",
            "stability_prob",
        ) if key in records[0]),
        None,
    )
    if stability_key is None:
        return None, None
    # Legacy classifier CSVs use the old convention that lower is better;
    # larger MegaScale ΔG target predictions rank higher.
    scored_records = []
    for row in records:
        try:
            row_score = float(row.get(stability_key, ""))
        except ValueError:
            continue
        scored_records.append((row_score, row))
    if not scored_records:
        return None, None
    reverse = stability_key != "stability_prob"
    sorted_recs = sorted(scored_records, key=lambda item: item[0], reverse=reverse)
    return sorted_recs[0][1], stability_key

def main():
    configs = ["baseline", "entropy", "ebm", "dual"]
    out_dir = Path("outputs/folded_structures")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("[*] Loading top sequences and submitting to ESMFold API...")
    results = {}
    for name in configs:
        csv_path = Path(f"outputs/ablation_{name}/design_library.csv")
        rec, stability_key = get_top_sequence(csv_path)
        if not rec:
            print(f"[-] No sequence found for {name}")
            continue
            
        aa_seq = rec["aa_seq"]
        score_label = (
            "predicted MegaScale assay ΔG target (kcal/mol)"
            if stability_key == "stability_megascale_delta_g_pred_kcal_mol"
            else "legacy stability probability"
        )
        print(f"[*] Folding {name} sequence (length={len(aa_seq)}, "
              f"{score_label}={rec.get(stability_key)})...")
        print(f"    Seq: {aa_seq[:40]}...")
        
        fold_res = esm_fold(aa_seq)
        if fold_res:
            results[name] = {
                "seq_id": rec["seq_id"],
                "aa_seq": aa_seq,
                "critic_stability_score": float(rec[stability_key]),
                "critic_stability_score_label": score_label,
                "plddt_mean": fold_res["plddt_mean"],
                "plddt_min": fold_res["plddt_min"],
                "plddt_max": fold_res["plddt_max"],
            }
            # Save PDB file
            pdb_path = out_dir / f"{name}_top_seq_{rec['seq_id']}.pdb"
            pdb_path.write_text(fold_res["pdb_text"])
            print(f"    [+] Success! Saved PDB to {pdb_path} (pLDDT: {fold_res['plddt_mean']:.2f})")
        else:
            print(f"    [-] ESMFold failed for {name}")
            
    # Print comparison table
    print("\n=== ESMFold STRUCTURE-CONFIDENCE RESULTS ===")
    print(f"{'Configuration':<15} | {'Sequence ID':<11} | {'Critic Score':<15} | {'Score Type':<45} | {'Mean pLDDT':<10} | {'Min pLDDT':<10} | {'Max pLDDT':<10}")
    print("-" * 150)
    for name, res in results.items():
        print(f"{name:<15} | {res['seq_id']:<11} | {res['critic_stability_score']:<15.4f} | {res['critic_stability_score_label']:<45} | {res['plddt_mean']:<10.2f} | {res['plddt_min']:<10.2f} | {res['plddt_max']:<10.2f}")

if __name__ == "__main__":
    main()
