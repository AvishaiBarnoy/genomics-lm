#!/usr/bin/env python3
"""
Optimize sequence embeddings using Langevin MCMC under the EBM model, decode, and evaluate pLDDT.
"""

import sys
import torch
from pathlib import Path
import csv

# Ensure src/ and scripts/ are on path
sys.path.append(str(Path(__file__).parent.parent))

from src.protein_lm.tokenizer import ProteinTokenizer
from src.protein_lm.ebm import ProteinLatentEBM
from src.protein_lm.sampler import latent_langevin_sample
from scripts.generative_design_loop import load_critic, esm_fold

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
    # Select sequence with lowest unstable probability
    sorted_recs = sorted(records, key=lambda x: float(x.get("stability_prob", 1.0)))
    return sorted_recs[0]

import torch.nn as nn

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[*] Using device: {device}")
    
    # 1. Load Critic
    critic_ckpt = "runs/2026-07-05_critic_bidirectional_attention_scaled/checkpoints/best_critic.pt"
    critic_cfg = "configs/protein_critic.yaml"
    print(f"[*] Loading critic model from {critic_ckpt}...")
    critic, tokenizer, task_dims = load_critic(critic_ckpt, critic_cfg, device)
    
    # 2. Load EBM
    ebm_ckpt = "runs/protein_ebm/checkpoints/best_ebm.pt"
    print(f"[*] Loading trained EBM model from {ebm_ckpt}...")
    # EBM latent dim matches bottleneck dim of the critic (256)
    ebm = ProteinLatentEBM(n_embd=256, hidden_dim=512).to(device)
    ebm_state = torch.load(ebm_ckpt, map_location=device)
    if "model" in ebm_state:
        ebm_state = ebm_state["model"]
    ebm.load_state_dict(ebm_state)
    ebm.eval()
    
    # 3. Load top initial sequence (from EBM early-abort sweep)
    csv_path = Path("outputs/ablation_ebm/design_library.csv")
    rec = get_top_sequence(csv_path)
    if not rec:
        print("[-] Error: No top sequence found from EBM sweep.")
        sys.exit(1)
        
    initial_seq = rec["aa_seq"]
    initial_stability_prob = float(rec["stability_prob"])
    print(f"\n[+] Loaded Initial Sequence (ID: {rec['seq_id']})")
    print(f"    Seq: {initial_seq}")
    print(f"    Critic Unstable Probability: {initial_stability_prob:.4f}")
    
    # 4. Perform Langevin Dynamics Optimization
    steps = 150
    lr = 15.0
    noise_std = 0.1
    lambda_reg = 0.01
    print(f"\n[*] Running Langevin Dynamics optimization for {steps} steps (lr={lr}, noise={noise_std}, lambda_reg={lambda_reg})...")
    
    optimized_seq, energy_history = latent_langevin_sample(
        ebm_model=ebm,
        critic_model=critic,
        tokenizer=tokenizer,
        initial_seq=initial_seq,
        steps=steps,
        lr=lr,
        noise_std=noise_std,
        lambda_reg=lambda_reg,
        temperature_reg=1.0,
        normalize_grad=True,
        device=device
    )
    
    print("[+] Optimization completed!")
    print(f"    Starting EBM Energy: {energy_history[0]:.4f}")
    print(f"    Final EBM Energy:    {energy_history[-1]:.4f}")
    print(f"    Energy Delta:        {energy_history[-1] - energy_history[0]:.4f}")
    
    # Calculate mutations
    mutations = sum(1 for a, b in zip(initial_seq, optimized_seq) if a != b)
    print(f"    [+] Sequence mutations introduced: {mutations} / {len(initial_seq)} residues ({mutations/len(initial_seq)*100:.1f}%)")
    
    # 5. Score optimized sequence under the critic
    from scripts.generative_design_loop import score_with_critic
    opt_crit = score_with_critic(critic, tokenizer, task_dims, optimized_seq, device)
    opt_stability_prob = opt_crit.get("stability_prob", 1.0)
    
    # Calculate vocabulary distance stats
    with torch.no_grad():
        vocab_embeds = critic.backbone.token_embedding.weight
        vocab_distances = []
        for i in range(len(tokenizer.amino_acids)):
            for j in range(i+1, len(tokenizer.amino_acids)):
                id_i = tokenizer.token_to_id[tokenizer.amino_acids[i]]
                id_j = tokenizer.token_to_id[tokenizer.amino_acids[j]]
                vocab_distances.append(torch.norm(vocab_embeds[id_i] - vocab_embeds[id_j]).item())
        avg_vocab_dist = sum(vocab_distances) / len(vocab_distances)
        
    print(f"\n[+] Embedding space statistics:")
    print(f"    Average distance between amino acid embeddings in vocab: {avg_vocab_dist:.4f}")
    
    print(f"\n[+] Optimized Sequence: {optimized_seq}")
    print(f"    Critic Unstable Probability: {opt_stability_prob:.4f}")
    
    # 6. Evaluate Physical Stability using ESMFold API
    print("\n[*] Folding initial vs optimized sequences using ESMFold API...")
    
    print("    [ESMFold] Folding initial sequence...")
    init_fold = esm_fold(initial_seq)
    if init_fold:
        print(f"    [+] Initial Sequence pLDDT: {init_fold['plddt_mean']:.2f}")
    else:
        print("    [-] Failed to fold initial sequence.")
        
    print("    [ESMFold] Folding optimized sequence...")
    opt_fold = esm_fold(optimized_seq)
    if opt_fold:
        print(f"    [+] Optimized Sequence pLDDT: {opt_fold['plddt_mean']:.2f}")
    else:
        print("    [-] Failed to fold optimized sequence.")
        
    # Compare
    print("\n=== LANGEVIN EMBEDDING OPTIMIZATION RESULTS ===")
    print(f"Metric                       | Initial Sequence    | Optimized Sequence  | Delta")
    print("-" * 85)
    print(f"EBM Energy Score             | {energy_history[0]:<19.4f} | {energy_history[-1]:<19.4f} | {energy_history[-1] - energy_history[0]:.4f}")
    print(f"Critic Prob (Unstable)       | {initial_stability_prob:<19.4f} | {opt_stability_prob:<19.4f} | {opt_stability_prob - initial_stability_prob:.4f}")
    
    init_plddt = init_fold['plddt_mean'] if init_fold else float('nan')
    opt_plddt = opt_fold['plddt_mean'] if opt_fold else float('nan')
    print(f"ESMFold Mean pLDDT           | {init_plddt:<19.2f} | {opt_plddt:<19.2f} | {opt_plddt - init_plddt:.2f}")

    # Save PDBs
    out_dir = Path("outputs/folded_structures")
    out_dir.mkdir(parents=True, exist_ok=True)
    if opt_fold:
        pdb_path = out_dir / "optimized_seq_langevin.pdb"
        pdb_path.write_text(opt_fold["pdb_text"])
        print(f"\n[+] Saved optimized structure PDB → {pdb_path}")

if __name__ == "__main__":
    main()
