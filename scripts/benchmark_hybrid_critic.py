#!/usr/bin/env python3
"""
Benchmark script for the Hybrid DNA-Protein Critic Guided Generation.
Generates sequences under various guidance settings (alpha) and compares
stability probability, EBM energy scores, ORF validity, and speed.
"""

import time
import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure src/ and scripts/ are on path
sys.path.append(str(Path(__file__).parent.parent))

from src.protein_lm.tokenizer import ProteinTokenizer
from src.protein_lm.ebm import ProteinLatentEBM
from src.codonlm.generate import generate_cds_critic_guided
from src.eval.inference_playground import load_codon_model, translate_codons_to_aa
from scripts.generative_design_loop import load_critic, score_with_critic

def run_benchmark():
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[*] Using device: {device}")

    # 1. Load generator (CodonLM)
    gen_run = "runs/2026-07-05_stage3_structured_pdb_replay_finetune"
    print(f"[*] Loading codon generator from {gen_run}...")
    gen_model, itos, stoi, _ = load_codon_model(gen_run)
    gen_model.to(device).eval()

    # 2. Load critic model
    critic_ckpt = "runs/2026-07-05_critic_bidirectional_attention_scaled/checkpoints/best_critic.pt"
    critic_cfg = "configs/protein_critic.yaml"
    print(f"[*] Loading critic model from {critic_ckpt}...")
    critic, c_tokenizer, task_dims = load_critic(critic_ckpt, critic_cfg, device)

    # 3. Load EBM model
    ebm_ckpt = "runs/protein_ebm/checkpoints/best_ebm.pt"
    print(f"[*] Loading trained EBM model from {ebm_ckpt}...")
    ebm = ProteinLatentEBM(n_embd=256, hidden_dim=512).to(device)
    ebm_state = torch.load(ebm_ckpt, map_location=device)
    if "model" in ebm_state:
        ebm_state = ebm_state["model"]
    ebm.load_state_dict(ebm_state)
    ebm.eval()

    # 4. Configurations to compare
    configs = [
        {"name": "Baseline (No guidance)", "alpha": 0.0, "task": "stability", "ebm": None},
        {"name": "Critic Guided (alpha=0.5)", "alpha": 0.5, "task": "stability", "ebm": None},
        {"name": "Critic Guided (alpha=2.0)", "alpha": 2.0, "task": "stability", "ebm": None},
        {"name": "EBM Guided (alpha=1.0)", "alpha": 1.0, "task": "ebm", "ebm": ebm},
        {"name": "EBM Guided (alpha=4.0)", "alpha": 4.0, "task": "ebm", "ebm": ebm},
    ]

    n_samples = 5
    target_codons = 60
    hard_cap = 100
    ctx_ids = [stoi.get("<BOS_CDS>", 1), stoi.get("ATG")]

    results = []

    for cfg in configs:
        print(f"\n[*] Evaluating: {cfg['name']}")
        cfg_results = []
        
        t0 = time.time()
        for idx in range(n_samples):
            # Generate guided sequence
            gen_ids, info = generate_cds_critic_guided(
                model=gen_model,
                critic_model=critic,
                c_tokenizer=c_tokenizer,
                device=device,
                ctx_ids=ctx_ids,
                stoi=stoi,
                itos=itos,
                target_codons=target_codons,
                hard_cap=hard_cap,
                alpha=cfg["alpha"],
                guide_top_k=5,
                target_task=cfg["task"],
                target_class_idx=0,
                ebm_model=cfg["ebm"],
                temperature=1.0,
                cds_only=True
            )
            
            # Decode codons
            codon_list = [itos[i] for i in gen_ids if len(itos[i]) == 3 and not (itos[i].startswith("<") or itos[i].endswith(">"))]
            raw_seq = "".join(codon_list)
            aa_seq = translate_codons_to_aa(codon_list)

            # Evaluate stability prob & EBM energy
            crit_scores = score_with_critic(critic, c_tokenizer, task_dims, aa_seq, device)
            stability_prob = crit_scores.get("stability_prob", 1.0)
            stable_prob = 1.0 - stability_prob

            # Extract EBM Energy
            with torch.no_grad():
                ids_c = [c_tokenizer.bos_token_id] + c_tokenizer.encode_sequence(aa_seq) + [c_tokenizer.eos_token_id]
                ids_t = torch.tensor([ids_c], dtype=torch.long, device=device)
                
                # Forward pass to EBM
                seq_length = ids_t.size(1)
                pos_embeds = critic.backbone.position_embedding(torch.arange(seq_length, device=device))
                x = critic.backbone.token_embedding(ids_t)
                x = critic.backbone.dropout(x + pos_embeds)
                is_causal = not getattr(critic.config, "bidirectional", True)
                causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_length, device=device) if is_causal else None
                for block in critic.backbone.transformer_blocks:
                    x = block(x, src_mask=causal_mask)
                if critic.pooling_type == "attention":
                    pooled, _ = critic.pooler(x)
                else:
                    pooled = x.mean(dim=1)
                latent = critic.shared_latent(pooled)
                energy = ebm(latent).item()

            # ORF validity
            has_stop_only_at_end = False
            stops = [i for i, t in enumerate(codon_list) if t in ["TAA", "TAG", "TGA"]]
            if stops:
                has_stop_only_at_end = (stops[-1] == len(codon_list) - 1) and (len(stops) == 1)
            
            cfg_results.append({
                "stable_prob": stable_prob,
                "energy": energy,
                "valid_orf": 1.0 if has_stop_only_at_end else 0.0,
                "tokens": len(gen_ids)
            })

        t_elapsed = time.time() - t0
        total_tokens = sum(r["tokens"] for r in cfg_results)
        tok_per_sec = total_tokens / t_elapsed if t_elapsed > 0 else 0.0

        mean_stable = np.mean([r["stable_prob"] for r in cfg_results])
        mean_energy = np.mean([r["energy"] for r in cfg_results])
        orf_rate = np.mean([r["valid_orf"] for r in cfg_results])

        results.append({
            "Configuration": cfg["name"],
            "Alpha Weight": cfg["alpha"],
            "Target Task": cfg["task"],
            "Stable Prob (Critic) ↑": mean_stable,
            "EBM Energy ↓": mean_energy,
            "Valid ORF Rate": orf_rate,
            "Speed (tokens/sec)": tok_per_sec
        })

    # 5. Format results as dataframe
    df_res = pd.DataFrame(results)
    print("\n=== HYBRID CRITIC GENERATION BENCHMARK RESULTS ===")
    print(df_res.to_string(index=False))

    # Save report
    out_dir = Path("outputs/benchmark_hybrid")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.md"
    
    with open(report_path, "w") as f:
        f.write("# 🧪 Hybrid DNA-Protein Critic Benchmark Report\n\n")
        f.write("Closed-loop bidirectional guided codon generation vs. standard sampling.\n\n")
        f.write(df_res.to_markdown(index=False))
        f.write("\n\n*Benchmark completed successfully.*")
    print(f"\n[+] Saved report to: {report_path}")

if __name__ == "__main__":
    run_benchmark()
