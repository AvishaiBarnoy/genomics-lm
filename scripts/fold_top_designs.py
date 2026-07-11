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

def get_top_sequence(csv_path: Path) -> dict:
    if not csv_path.exists():
        return None
    records = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    if not records:
        return None
    # Sort by stability score (index 0 corresponds to stable)
    # Since scores["stability_prob"] = stab_probs[-1] (index 1 = unstable)
    # the most stable sequence is the one with the MINIMUM scores["stability_prob"]!
    sorted_recs = sorted(records, key=lambda x: float(x.get("stability_prob", 1.0)))
    return sorted_recs[0]

def main():
    configs = ["baseline", "entropy", "ebm", "dual"]
    out_dir = Path("outputs/folded_structures")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("[*] Loading top sequences and submitting to ESMFold API...")
    results = {}
    for name in configs:
        csv_path = Path(f"outputs/ablation_{name}/design_library.csv")
        rec = get_top_sequence(csv_path)
        if not rec:
            print(f"[-] No sequence found for {name}")
            continue
            
        aa_seq = rec["aa_seq"]
        print(f"[*] Folding {name} sequence (length={len(aa_seq)}, stability_prob={rec.get('stability_prob')})...")
        print(f"    Seq: {aa_seq[:40]}...")
        
        fold_res = esm_fold(aa_seq)
        if fold_res:
            results[name] = {
                "seq_id": rec["seq_id"],
                "aa_seq": aa_seq,
                "stability_prob": float(rec["stability_prob"]),
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
    print("\n=== PHYSICAL STABILITY VALUES (ESMFold pLDDT) ===")
    print(f"{'Configuration':<15} | {'Sequence ID':<11} | {'Stability Prob (Unstable)':<25} | {'Mean pLDDT':<10} | {'Min pLDDT':<10} | {'Max pLDDT':<10}")
    print("-" * 92)
    for name, res in results.items():
        print(f"{name:<15} | {res['seq_id']:<11} | {res['stability_prob']:<25.4f} | {res['plddt_mean']:<10.2f} | {res['plddt_min']:<10.2f} | {res['plddt_max']:<10.2f}")

if __name__ == "__main__":
    main()
