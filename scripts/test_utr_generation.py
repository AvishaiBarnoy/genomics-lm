#!/usr/bin/env python3
"""
Test if CodonLM generates secondary structure termination motifs (hairpins, poly-T tracts)
specifically when it is prompted past the STOP codon into the 3' UTR region.
"""

import argparse
from pathlib import Path
import numpy as np
import torch

from scripts._shared import resolve_run, load_token_list, stoi
from src.codonlm.generate import generate_cds_constrained
from scripts.check_termination_motifs import find_hairpin_score, get_max_t_run

STOP_CODONS = {"TAA", "TAG", "TGA"}

def run_utr_generation_test(run_id: str, ckpt: str = "best.pt", n_samples: int = 50):
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
        
    ckpt_dict = torch.load(weights_path, map_location=device)
    state_dict = ckpt_dict["model"] if isinstance(ckpt_dict, dict) and "model" in ckpt_dict else ckpt_dict
    model = build_model(spec_obj)
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    
    tokens = load_token_list(run_dir)
    tok_stoi = stoi(tokens)
    
    # Load raw DNA dataset
    manifest_path = run_dir / "combined_manifest.json"
    manifest = eval(manifest_path.read_text())
    dna_file_path = Path(manifest["datasets"][0]["dna"])
    if not dna_file_path.is_absolute():
        dna_file_path = repo / dna_file_path
        
    print(f"[*] Extracting genes and 3' UTR boundaries from {dna_file_path}...")
    
    # We want to find cases in the raw dataset where a stop codon is present,
    # and we have at least 30 bp (10 codons) of sequence following it.
    stop_utr_pairs = []
    
    with open(dna_file_path) as f:
        for line in f:
            seq = line.strip().upper().replace("U", "T")
            # Scan for a stop codon followed by enough nucleotides
            for i in range(30, len(seq) - 33, 3):
                codon = seq[i:i+3]
                if codon in STOP_CODONS:
                    prefix = seq[i-30:i+3] # 10 codons leading up to and including the stop
                    utr = seq[i+3:i+33]    # 30 bp of biological UTR
                    stop_utr_pairs.append((prefix, utr))
                    break
            if len(stop_utr_pairs) >= n_samples:
                break
                
    print(f"[*] Found {len(stop_utr_pairs)} stop-UTR pairs. Generating continuations...")
    
    biological_utrs = []
    generated_utrs = []
    control_utrs = [] # generations from a mid-CDS prefix
    
    bos_id = tok_stoi.get("<BOS_CDS>")
    
    for idx, (prefix, real_utr) in enumerate(stop_utr_pairs):
        # 1. Generate from Stop-terminated prefix
        ctx_ids = []
        if bos_id is not None:
            ctx_ids.append(bos_id)
        for i in range(0, len(prefix), 3):
            codon = prefix[i:i+3]
            if codon in tok_stoi:
                ctx_ids.append(tok_stoi[codon])
                
        # Generate next 10 codons
        gen_ids, _ = generate_cds_constrained(
            model=model, device=device, ctx_ids=ctx_ids, stoi=tok_stoi, itos=tokens,
            target_codons=10, hard_cap=10, require_terminal_stop=False,
            temperature=1.0, topk=0
        )
        gen_tokens = [tokens[i] for i in gen_ids]
        codons = [t for t in gen_tokens if len(t) == 3 and set(t) <= set("ACGT")]
        # Extract only the newly generated 10 codons (30bp)
        dna_gen = "".join(codons)[len(prefix):]
        if len(dna_gen) >= 30:
            generated_utrs.append(dna_gen[:30])
            biological_utrs.append(real_utr)
            
        # 2. Generate from Mid-CDS control prefix (no stop codon)
        mid_prefix = prefix[:-3] + "GCG" # replace stop codon with Ala
        ctx_ids_ctrl = []
        if bos_id is not None:
            ctx_ids_ctrl.append(bos_id)
        for i in range(0, len(mid_prefix), 3):
            codon = mid_prefix[i:i+3]
            if codon in tok_stoi:
                ctx_ids_ctrl.append(tok_stoi[codon])
                
        gen_ids_ctrl, _ = generate_cds_constrained(
            model=model, device=device, ctx_ids=ctx_ids_ctrl, stoi=tok_stoi, itos=tokens,
            target_codons=10, hard_cap=10, require_terminal_stop=False,
            temperature=1.0, topk=0
        )
        gen_tokens_ctrl = [tokens[i] for i in gen_ids_ctrl]
        codons_ctrl = [t for t in gen_tokens_ctrl if len(t) == 3 and set(t) <= set("ACGT")]
        dna_gen_ctrl = "".join(codons_ctrl)[len(mid_prefix):]
        if len(dna_gen_ctrl) >= 30:
            control_utrs.append(dna_gen_ctrl[:30])

    print(f"\n[*] Evaluating structural properties of UTR regions (N={len(generated_utrs)}):")
    
    def get_stats(seq_list):
        hairpin_scores = [find_hairpin_score(s) for s in seq_list]
        poly_t_runs = [get_max_t_run(s) for s in seq_list]
        return {
            "hairpin_avg": float(np.mean(hairpin_scores)),
            "hairpin_max": float(np.max(hairpin_scores)),
            "poly_t_avg": float(np.mean(poly_t_runs)),
            "poly_t_max": int(np.max(poly_t_runs)),
            "poly_t_gte_4_frac": float(sum(1 for r in poly_t_runs if r >= 4) / len(poly_t_runs)),
        }
        
    bio_stats = get_stats(biological_utrs)
    gen_stats = get_stats(generated_utrs)
    ctrl_stats = get_stats(control_utrs)
    
    print("\n=== UTR Generation Test Results ===")
    print("Biological Ground Truth UTR:")
    print(f"  - Avg Hairpin Score: {bio_stats['hairpin_avg']:.2f} (max={bio_stats['hairpin_max']})")
    print(f"  - Avg Poly-T Run: {bio_stats['poly_t_avg']:.2f} (max={bio_stats['poly_t_max']})")
    print(f"  - Fraction of sequences with Poly-T run >= 4: {bio_stats['poly_t_gte_4_frac']*100:.1f}%")
    
    print("\nGenerated Post-STOP UTR (Model-generated):")
    print(f"  - Avg Hairpin Score: {gen_stats['hairpin_avg']:.2f} (max={gen_stats['hairpin_max']})")
    print(f"  - Avg Poly-T Run: {gen_stats['poly_t_avg']:.2f} (max={gen_stats['poly_t_max']})")
    print(f"  - Fraction of sequences with Poly-T run >= 4: {gen_stats['poly_t_gte_4_frac']*100:.1f}%")
    
    print("\nControl Mid-CDS Generations:")
    print(f"  - Avg Hairpin Score: {ctrl_stats['hairpin_avg']:.2f} (max={ctrl_stats['hairpin_max']})")
    print(f"  - Avg Poly-T Run: {ctrl_stats['poly_t_avg']:.2f} (max={ctrl_stats['poly_t_max']})")
    print(f"  - Fraction of sequences with Poly-T run >= 4: {ctrl_stats['poly_t_gte_4_frac']*100:.1f}%")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="2026-06-06_stage2.5_6L4H_d256_e20")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--samples", type=int, default=50)
    args = ap.parse_args()
    
    run_utr_generation_test(args.run_id, args.ckpt, args.samples)
