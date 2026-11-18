#!/usr/bin/env python3
"""
Evaluate protein prediction scores and biological properties of generated sequences
using the corrected ProteinCritic (family=43 Pfam classes, function=7 EC classes,
stability=regression ΔG kcal/mol).

Metrics:
  - Natural stop rate / hard-cap rate
  - Mean generated AA length
  - AA sequence identity vs. ground-truth continuation
  - ProteinCritic: stability ΔG (kcal/mol, lower ≈ more stable)
  - ProteinCritic: family top-1 confidence (Pfam)
  - ProteinCritic: function top-1 confidence (EC)

Models compared:
  1. Baseline (locked primary): corrected-codonlm-v1-batch64-lr-ablation-lr_1_5e4
  2. Calibrated-only (1.12× head.weight): corrected-primary-weight-space-programming-seed1337
  3. Calibrated + Steering + Deflation (aggressive): corrected-aggressive-weight-space-programming-seed1337
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch

from src.codonlm.checkpoints import build_codon_model_from_cfg, load_codon_checkpoint
from src.codonlm.generate import generate_model_raw
from scripts.generative_design_loop import load_critic, score_with_critic

CODON_TABLE = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
    'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
    'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
    'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
    'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
    'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
    'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
    'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'*', 'TAG':'*',
    'TGC':'C', 'TGT':'C', 'TGA':'*', 'TGG':'W',
}


def translate_codons(codons: list[str]) -> str:
    aa = []
    for c in codons:
        a = CODON_TABLE.get(c, "X")
        if a == "*":
            break
        aa.append(a)
    return "".join(aa)


def aa_identity(truth_aa: str, gen_aa: str) -> float:
    L = min(len(truth_aa), len(gen_aa))
    if L == 0:
        return 0.0
    return sum(1 for i in range(L) if truth_aa[i] == gen_aa[i]) / L


def evaluate_models_on_protein_scores(
    baseline_run: str = "runs/corrected-codonlm-v1-batch64-lr-ablation-lr_1_5e4",
    critic_ckpt: str = "runs/corrected-protein-critic-v1-seed1337/checkpoints/last_critic.pt",
    critic_cfg: str = "configs/corrected_protein_critic_v1.yaml",
    n_genes: int = 20,
    max_new: int = 100,
    k: int = 3,
):
    device = torch.device("cpu")  # Low RAM usage (< 250 MB)

    # 1. Load corrected ProteinCritic (family=43, function=7, stability=regression)
    print("Loading corrected ProteinCritic (family/function/stability) on CPU...", flush=True)
    critic_model, critic_tokenizer, critic_task_dims = load_critic(
        critic_ckpt, critic_cfg, device
    )
    print(f"  Task dims: {critic_task_dims}", flush=True)
    print(f"  Regression tasks: {list(getattr(critic_model, 'regression_tasks', []))}", flush=True)

    # 2. Load vocab from baseline run
    with open(f"{baseline_run}/itos.txt") as f:
        itos = [line.strip() for line in f]
    stoi = {tok: i for i, tok in enumerate(itos)}

    # 3. Load test genes
    genes_file = Path("runs/corrected-multi-offset-heads-seed1337/scores/_eval_test_cds_dna.txt")
    with open(genes_file) as f:
        test_genes = [line.strip() for line in f if len(line.strip()) >= 60][:n_genes]
    print(f"  Loaded {len(test_genes)} test genes (k={k}, max_new={max_new}).", flush=True)

    # Models to evaluate: label -> run_dir
    runs = {
        "Baseline (Locked Primary)":
            "runs/corrected-codonlm-v1-batch64-lr-ablation-lr_1_5e4",
        "Calibrated-only (1.12× head.weight)":
            "runs/corrected-primary-weight-space-programming-seed1337",
        "Calibrated + Steering + Deflation (aggressive)":
            "runs/corrected-aggressive-weight-space-programming-seed1337",
    }

    results = {}

    for model_name, run_dir in runs.items():
        print(f"\n{'='*60}", flush=True)
        print(f"Evaluating: {model_name}", flush=True)
        print(f"  Run dir: {run_dir}", flush=True)

        state, cfg, _ = load_codon_checkpoint(run_dir)
        model = build_codon_model_from_cfg(cfg)
        model.load_state_dict(state)
        model.to(device)
        model.eval()

        delta_gs = []           # stability regression (kcal/mol)
        family_confs = []       # top-1 Pfam confidence
        function_confs = []     # top-1 EC confidence
        family_entropies = []   # Pfam entropy (spread)
        function_entropies = [] # EC entropy (spread)
        identities = []
        stop_count = 0
        hard_cap_count = 0
        lengths = []

        for idx, g in enumerate(test_genes):
            truth_codons = [g[i:i+3] for i in range(0, len(g), 3) if len(g[i:i+3]) == 3]
            truth_aa = translate_codons(truth_codons)

            prefix_codons = truth_codons[:k]
            ctx_ids = [stoi["<BOS_CDS>"]] + [stoi[c] for c in prefix_codons if c in stoi]

            with torch.no_grad():
                gen_ids, meta = generate_model_raw(
                    model, device, ctx_ids, stoi, itos,
                    max_new_tokens=max_new, temperature=1.0, topk=0
                )

            if meta["had_terminal_stop"]:
                stop_count += 1
            if meta["hit_hard_cap"]:
                hard_cap_count += 1

            gen_codons = [itos[cid] for cid in gen_ids[len(ctx_ids):]
                          if len(itos[cid]) == 3]
            gen_aa = translate_codons(gen_codons)
            lengths.append(len(gen_aa))

            truth_cont_aa = truth_aa[k:]
            identities.append(aa_identity(truth_cont_aa, gen_aa))

            if len(gen_aa) >= 5:
                crit = score_with_critic(
                    critic_model, critic_tokenizer, critic_task_dims, gen_aa, device
                )
                # Stability: regression ΔG (kcal/mol)
                if "stability_megascale_delta_g_pred_kcal_mol" in crit:
                    delta_gs.append(crit["stability_megascale_delta_g_pred_kcal_mol"])
                # Family: top-1 Pfam confidence
                if "family_top1_conf" in crit:
                    family_confs.append(crit["family_top1_conf"])
                if "family_entropy" in crit:
                    family_entropies.append(crit["family_entropy"])
                # Function: top-1 EC confidence
                if "function_top1_conf" in crit:
                    function_confs.append(crit["function_top1_conf"])
                if "function_entropy" in crit:
                    function_entropies.append(crit["function_entropy"])

            if (idx + 1) % 5 == 0:
                print(f"  [{idx+1}/{len(test_genes)}] stop={stop_count} "
                      f"ΔG_mean={np.mean(delta_gs):.3f}" if delta_gs
                      else f"  [{idx+1}/{len(test_genes)}] stop={stop_count}",
                      flush=True)

        results[model_name] = {
            "stop_rate": stop_count / len(test_genes),
            "hard_cap_rate": hard_cap_count / len(test_genes),
            "mean_aa_len": float(np.mean(lengths)),
            "mean_aa_identity": float(np.mean(identities)),
            "mean_delta_g_kcal_mol": float(np.mean(delta_gs)) if delta_gs else None,
            "mean_family_top1_conf": float(np.mean(family_confs)) if family_confs else None,
            "mean_family_entropy": float(np.mean(family_entropies)) if family_entropies else None,
            "mean_function_top1_conf": float(np.mean(function_confs)) if function_confs else None,
            "mean_function_entropy": float(np.mean(function_entropies)) if function_entropies else None,
            "n_critic_scored": len(delta_gs),
        }

    # Print summary table
    print("\n" + "=" * 70)
    print("FINAL PROTEIN PREDICTION SCORE COMPARISON (Corrected Critic)")
    print("=" * 70)
    header = f"{'Metric':<40} {'Baseline':>12} {'Calib':>12} {'Aggressive':>12}"
    print(header)
    print("-" * 76)

    model_keys = list(results.keys())

    def row(label, key, fmt="{:.4f}", scale=1.0):
        vals = [results[k].get(key) for k in model_keys]
        cells = [fmt.format(v * scale) if v is not None else "     N/A" for v in vals]
        print(f"{label:<40} {cells[0]:>12} {cells[1]:>12} {cells[2]:>12}")

    row("Natural Stop Rate (%)",      "stop_rate",              fmt="{:.1f}", scale=100)
    row("Hard-Cap Rate (%)",           "hard_cap_rate",          fmt="{:.1f}", scale=100)
    row("Mean AA Length",              "mean_aa_len",            fmt="{:.1f}")
    row("Mean AA Identity (%)",        "mean_aa_identity",       fmt="{:.2f}", scale=100)
    row("Mean ΔG pred (kcal/mol)",     "mean_delta_g_kcal_mol",  fmt="{:.3f}")
    row("Family Top-1 Conf (Pfam)",    "mean_family_top1_conf",  fmt="{:.4f}")
    row("Family Entropy (nats)",       "mean_family_entropy",    fmt="{:.3f}")
    row("Function Top-1 Conf (EC)",    "mean_function_top1_conf",fmt="{:.4f}")
    row("Function Entropy (nats)",     "mean_function_entropy",  fmt="{:.3f}")
    print(f"\n  Note: ΔG < 0 ⟹ predicted stable; Family/Function conf ↑ ⟹ more confident fold/function assignment")
    print(f"  Critic: runs/corrected-protein-critic-v1-seed1337 "
          f"(Pfam={critic_task_dims.get('family',0)} classes, "
          f"EC={critic_task_dims.get('function',0)} classes, "
          f"stability=regression)")

    # Also dump JSON
    out_path = Path("runs/_summary/wsp_protein_eval_corrected_critic.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    return results


if __name__ == "__main__":
    evaluate_models_on_protein_scores()
