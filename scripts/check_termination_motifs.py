#!/usr/bin/env python3
"""
Script to analyze if the model has learned physical/structural termination cues
(like GC-rich hairpins and Poly-T tracts) without or before learning precise stop-codon placement.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch

from scripts._shared import load_model, resolve_run, load_token_list, stoi
from src.codonlm.generate import generate_cds_constrained
from scripts.probe_structural_awareness import get_theoretical_shape

STOP_CODONS = {"TAA", "TAG", "TGA"}

def find_hairpin_score(dna_seq: str) -> float:
    """
    Computes a heuristic score for hairpin/stem-loop stability.
    Matches A-T (score 2) and G-C (score 3), penalizes mismatch (score -1).
    Looks for the best complementary match in the sequence with a loop size of 3-9 nt.
    """
    n = len(dna_seq)
    pairs = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    best_score = 0.0
    
    # Iterate over possible loop lengths
    for loop in range(3, 10):
        # Iterate over starting position of the first stem strand
        for i in range(n):
            # Iterate over stem lengths
            for stem in range(3, 12):
                if i + 2 * stem + loop > n:
                    break
                s1 = dna_seq[i : i + stem]
                s2 = dna_seq[i + stem + loop : i + 2 * stem + loop][::-1]
                score = 0
                for a, b in zip(s1, s2):
                    if pairs.get(a) == b:
                        score += 3 if (a in 'GC' and b in 'GC') else 2
                    else:
                        score -= 1
                if score > best_score:
                    best_score = float(score)
    return best_score

def get_max_t_run(dna_seq: str) -> int:
    """
    Returns the length of the longest consecutive run of Ts (or Us) in the sequence.
    """
    max_run = 0
    current_run = 0
    for char in dna_seq:
        if char in ('T', 'U'):
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    return max_run

def analyze_termination_motifs(run_id: str, ckpt: str = "best.pt", n_samples: int = 100):
    run_id, run_dir = resolve_run(run_id=run_id)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[*] Loading model for run {run_id} on {device}...")
    from scripts._shared import read_meta, build_model, ModelSpec
    meta = read_meta(run_dir)
    spec_obj = ModelSpec.from_dict(meta["model_spec"])
    
    repo = Path(__file__).resolve().parents[1]
    weights_path = run_dir / "checkpoints" / ckpt
    if not weights_path.exists():
        weights_path = run_dir / ckpt
    if not weights_path.exists():
        weights_path = repo / "outputs" / "checkpoints" / run_id / ckpt
        
    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint {ckpt} not found for run {run_id}")
        
    print(f"[*] Loading weights from {weights_path}...")
    ckpt_dict = torch.load(weights_path, map_location=device)
    state_dict = ckpt_dict["model"] if isinstance(ckpt_dict, dict) and "model" in ckpt_dict else ckpt_dict
    model = build_model(spec_obj)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    tokens = load_token_list(run_dir)
    tok_stoi = stoi(tokens)
    
    # Load dataset to get prefixes
    manifest_path = run_dir / "combined_manifest.json"
    if not manifest_path.exists():
        print(f"[!] Manifest not found for run {run_id}")
        return
    
    import json
    manifest = json.loads(manifest_path.read_text())
    dna_file_path = Path(manifest["datasets"][0]["dna"])
    if not dna_file_path.is_absolute():
        dna_file_path = Path(__file__).resolve().parents[1] / dna_file_path
        
    if not dna_file_path.exists():
        print(f"[!] DNA dataset not found at {dna_file_path}")
        return
        
    print(f"[*] Reading prefixes from {dna_file_path}...")
    prefixes = []
    with open(dna_file_path) as f:
        for line in f:
            seq = line.strip().upper().replace("U", "T")
            # We want sequences long enough to extract a 10-codon (30 bp) prefix
            if len(seq) >= 60:
                prefixes.append(seq[:30])
            if len(prefixes) >= n_samples:
                break
                
    print(f"[*] Running generations for {len(prefixes)} prefixes...")
    early_term_seqs = []
    hard_capped_seqs = []
    
    # Token IDs for special tokens
    bos_id = tok_stoi.get("<BOS_CDS>")
    
    for idx, pref in enumerate(prefixes):
        ctx_ids = []
        if bos_id is not None:
            ctx_ids.append(bos_id)
        for i in range(0, len(pref), 3):
            codon = pref[i:i+3]
            if codon in tok_stoi:
                ctx_ids.append(tok_stoi[codon])
                
        # Generate continuation
        gen_ids, info = generate_cds_constrained(
            model=model,
            device=device,
            ctx_ids=ctx_ids,
            stoi=tok_stoi,
            itos=tokens,
            target_codons=60,
            hard_cap=150,
            require_terminal_stop=False,
            temperature=1.0,
            topk=0
        )
        
        # Convert generated IDs to DNA sequence
        gen_tokens = [tokens[i] for i in gen_ids]
        # Skip special tokens at the start
        codons = [t for t in gen_tokens if len(t) == 3 and set(t) <= set("ACGT")]
        dna_gen = "".join(codons)
        
        # We look at the window immediately preceding the termination point or end of generation
        # If the model terminated early, it stops at info["generated_codons"].
        # Since the sequence always has the 30bp prefix plus whatever was generated, its length is always >= 30bp.
        if len(dna_gen) < 30:
            continue
            
        analysis_window = dna_gen[-30:]
        
        is_early = info["generated_codons"] < 60
        
        if idx < 5:
            print(f"[Debug Sample {idx}] Generated: {info['generated_codons']} codons | Window: {analysis_window} | Early: {is_early}")
            
        if is_early:
            early_term_seqs.append(analysis_window)
        else:
            hard_capped_seqs.append(analysis_window)
            
    print(f"\n[*] Generation complete:")
    print(f"    - Early terminated: {len(early_term_seqs)} ({len(early_term_seqs)/(len(early_term_seqs)+len(hard_capped_seqs)+1e-9)*100:.1f}%)")
    print(f"    - Hard capped (reached target): {len(hard_capped_seqs)}")
    
    def extract_metrics(seq_list, name):
        hairpin_scores = []
        poly_t_runs = []
        mgw_means = []
        roll_means = []
        ep_means = []
        
        for s in seq_list:
            # We look at the last 30 bp of the generated sequence (where termination happened or generation ended)
            window = s[-30:] if len(s) >= 30 else s
            
            # Hairpin score
            hairpin_scores.append(find_hairpin_score(window))
            
            # Poly-T run
            poly_t_runs.append(get_max_t_run(window))
            
            # DNAshape properties
            shape = get_theoretical_shape(window)
            mgw_means.append(float(shape["MGW"].mean()))
            roll_means.append(float(shape["Roll"].mean()))
            ep_means.append(float(shape["EP"].mean()))
            
        if not seq_list:
            return None
            
        return {
            "count": len(seq_list),
            "hairpin_avg": float(np.mean(hairpin_scores)),
            "hairpin_max": float(np.max(hairpin_scores)),
            "poly_t_avg": float(np.mean(poly_t_runs)),
            "poly_t_max": int(np.max(poly_t_runs)),
            "poly_t_gte_4_frac": float(sum(1 for r in poly_t_runs if r >= 4) / len(poly_t_runs)),
            "mgw_avg": float(np.mean(mgw_means)),
            "roll_avg": float(np.mean(roll_means)),
            "ep_avg": float(np.mean(ep_means)),
        }
        
    early_metrics = extract_metrics(early_term_seqs, "Early Terminated")
    capped_metrics = extract_metrics(hard_capped_seqs, "Hard Capped")
    
    print("\n=== Analysis Results ===")
    if early_metrics:
        print(f"Early Terminated Seqs (N={early_metrics['count']}):")
        print(f"  - Avg Hairpin Score (GC-richness/symmetry): {early_metrics['hairpin_avg']:.2f} (max={early_metrics['hairpin_max']:.1f})")
        print(f"  - Avg Poly-T Run length: {early_metrics['poly_t_avg']:.2f} (max={early_metrics['poly_t_max']})")
        print(f"  - Fraction of sequences with Poly-T run >= 4: {early_metrics['poly_t_gte_4_frac']*100:.1f}%")
        print(f"  - DNAshape: MGW={early_metrics['mgw_avg']:.2f} Å, Roll={early_metrics['roll_avg']:.2f}°, EP={early_metrics['ep_avg']:.2f} kT/e")
        
    if capped_metrics:
        print(f"Hard Capped Seqs (N={capped_metrics['count']}):")
        print(f"  - Avg Hairpin Score (GC-richness/symmetry): {capped_metrics['hairpin_avg']:.2f} (max={capped_metrics['hairpin_max']:.1f})")
        print(f"  - Avg Poly-T Run length: {capped_metrics['poly_t_avg']:.2f} (max={capped_metrics['poly_t_max']})")
        print(f"  - Fraction of sequences with Poly-T run >= 4: {capped_metrics['poly_t_gte_4_frac']*100:.1f}%")
        print(f"  - DNAshape: MGW={capped_metrics['mgw_avg']:.2f} Å, Roll={capped_metrics['roll_avg']:.2f}°, EP={capped_metrics['ep_avg']:.2f} kT/e")
        
    if early_metrics and capped_metrics:
        hairpin_diff = early_metrics['hairpin_avg'] - capped_metrics['hairpin_avg']
        poly_t_diff = early_metrics['poly_t_avg'] - capped_metrics['poly_t_avg']
        print("\n--- Differences (Early Terminated - Hard Capped) ---")
        print(f"  - Hairpin Score difference: {hairpin_diff:+.2f}")
        print(f"  - Poly-T Run diff: {poly_t_diff:+.2f}")
        print(f"  - Poly-T >= 4 diff: {early_metrics['poly_t_gte_4_frac'] - capped_metrics['poly_t_gte_4_frac']:+.2%}")
        
    # Write report JSON
    out_dir = run_dir / "motif_mining"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_data = {
        "early_metrics": early_metrics,
        "capped_metrics": capped_metrics,
        "n_samples": n_samples
    }
    with open(out_dir / "termination_motifs_analysis.json", "w") as f:
        import json
        json.dump(report_data, f, indent=4)
    print(f"\n[success] Report saved to {out_dir / 'termination_motifs_analysis.json'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="2026-06-06_stage2.5_6L4H_d256_e20")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--samples", type=int, default=100)
    args = ap.parse_args()
    
    analyze_termination_motifs(args.run_id, args.ckpt, args.samples)
