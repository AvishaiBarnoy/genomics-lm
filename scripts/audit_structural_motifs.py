import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scripts.probe_structural_awareness import get_theoretical_shape

def audit_motifs(run_id):
    run_dir = Path("runs") / run_id
    motif_path = run_dir / "motif_mining" / "motif_report.md"
    
    if not motif_path.exists():
        print(f"[!] No motif report found for {run_id}")
        return

    # Extract consensus sequences
    motifs = []
    content = motif_path.read_text()
    for line in content.splitlines():
        if line.startswith("## Cluster"):
            name = line.split("## ")[1].split("(")[0].strip()
            motifs.append({"name": name})
        if line.startswith("**Consensus:**") and motifs:
            # Consensus: `SEQ`
            seq = line.split("`")[1].replace(" ", "").replace("<pad>", "")
            motifs[-1]["sequence"] = seq

    results = []
    print(f"=== Structural Motif Audit: {run_id} ===")
    
    for m in motifs:
        if not m.get("sequence"): continue
        
        # Calculate physical properties of the consensus DNA
        props = get_theoretical_shape(m["sequence"])
        
        # Average properties across the motif
        m_stats = {k: round(float(v.mean()), 2) for k, v in props.items()}
        
        # Interpretation
        features = []
        if m_stats["Roll"] > 4.0: features.append("Highly Flexible/Bent")
        elif m_stats["Roll"] < 1.0: features.append("Stiff/Straight")
        
        if m_stats["MGW"] < 3.8: features.append("Narrow Minor Groove")
        elif m_stats["MGW"] > 5.2: features.append("Wide Minor Groove")
        
        if m_stats["EP"] < -8.0: features.append("High Negative Charge (Attractive)")
        
        m["structural_stats"] = m_stats
        m["interpretations"] = features
        
        print(f"\n{m['name']} ({m['sequence']})")
        print(f"  Stats: {m_stats}")
        print(f"  Biological Role: {', '.join(features) if features else 'Standard B-DNA'}")
        
    # Save results
    out_path = run_dir / "motif_mining" / "structural_motif_audit.json"
    with open(out_path, "w") as f:
        json.dump(motifs, f, indent=4)
    print(f"\n[success] Structural audit saved to {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    args = ap.parse_args()
    audit_motifs(args.run_id)
