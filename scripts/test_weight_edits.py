#!/usr/bin/env python3
"""
Test weight-space interventions on CodonLM.
Evaluates:
1. Validation NLL / PPL on 100 sequences (batch_size=4, low memory, CPU or MPS)
2. Generation on 10 test gene prefixes (natural stop rate, hard cap rate, length)
3. Greedy repetition test (does greedy decoding break out of the CTG loop?)
"""

import copy
import math
import numpy as np
import torch
import torch.nn.functional as F

from src.codonlm.checkpoints import build_codon_model_from_cfg, load_codon_checkpoint
from src.codonlm.generate import generate_model_raw


def get_baseline_and_data(run_dir="runs/corrected-multi-offset-heads-seed1337"):
    state, cfg, _ = load_codon_checkpoint(run_dir)
    model = build_codon_model_from_cfg(cfg)
    model.load_state_dict(state)
    device = torch.device("cpu") # Keep strictly under 200MB RAM
    model.to(device)
    model.eval()

    with open(f"{run_dir}/itos.txt") as f:
        itos = [line.strip() for line in f]
    stoi = {tok: i for i, tok in enumerate(itos)}

    # Load 50 validation sequences for fast, low-memory evaluation
    val_data = np.load("data/processed/corrected/corrected-codonlm-v1/genome/val_bs512.npz")
    val_x = torch.from_numpy(val_data["X"][:50]).long().to(device)
    val_y = torch.from_numpy(val_data["Y"][:50]).long().to(device)

    # Load test genes for generation evaluation
    with open(f"{run_dir}/scores/_eval_test_cds_dna.txt") as f:
        genes = [line.strip() for line in f if len(line.strip()) >= 30][:10]

    return model, cfg, itos, stoi, val_x, val_y, genes, device


@torch.no_grad()
def eval_nll(model, val_x, val_y, batch_size=5):
    total_loss = 0.0
    total_tokens = 0
    for i in range(0, len(val_x), batch_size):
        bx = val_x[i:i+batch_size]
        by = val_y[i:i+batch_size]
        logits, _ = model(bx)
        flat_logits = logits.reshape(-1, logits.size(-1))
        flat_targets = by.reshape(-1)
        valid = (flat_targets != 0).sum().item()
        if valid == 0:
            continue
        loss = F.cross_entropy(flat_logits, flat_targets, ignore_index=0, reduction="sum")
        total_loss += loss.item()
        total_tokens += valid
    mean_nll = total_loss / total_tokens
    ppl = math.exp(mean_nll)
    return mean_nll, ppl


@torch.no_grad()
def eval_generation(model, device, stoi, itos, genes, max_new=100, k=3):
    stops = 0
    hard_caps = 0
    lengths = []
    gc_contents = []

    for g in genes:
        codons = [g[i:i+3] for i in range(0, min(len(g), 30), 3)]
        ctx_ids = [stoi["<BOS_CDS>"]] + [stoi[c] for c in codons[:k]]
        gen_ids, meta = generate_model_raw(
            model, device, ctx_ids, stoi, itos, max_new_tokens=max_new, temperature=1.0, topk=0
        )
        if meta["had_terminal_stop"]:
            stops += 1
        if meta["hit_hard_cap"]:
            hard_caps += 1
        lengths.append(meta["generated_codons"])

        gen_codons = [itos[cid] for cid in gen_ids[len(ctx_ids):] if len(itos[cid]) == 3]
        if gen_codons:
            dna = "".join(gen_codons)
            gc = (dna.count("G") + dna.count("C")) / len(dna)
            gc_contents.append(gc)

    return {
        "stop_rate": stops / len(genes),
        "hard_cap_rate": hard_caps / len(genes),
        "mean_len": float(np.mean(lengths)),
        "mean_gc": float(np.mean(gc_contents)) if gc_contents else 0.0,
    }


@torch.no_grad()
def eval_greedy_ctg_loop(model, device, stoi, itos):
    """Check whether greedy continuation from a prefix breaks the CTG loop."""
    prefix = ["<BOS_CDS>", "ATG", "CTT", "AGA", "CTG", "CTG"]
    ctx_ids = [stoi[t] for t in prefix]
    ids = list(ctx_ids)
    for _ in range(30):
        x = torch.tensor(ids[-512:], dtype=torch.long, device=device).unsqueeze(0)
        logits, _ = model(x)
        next_id = torch.argmax(logits[0, -1]).item()
        ids.append(next_id)
    gen = [itos[i] for i in ids[len(ctx_ids):]]
    ctg_count = gen.count("CTG")
    is_loop = ctg_count >= 20
    return is_loop, " ".join(gen[:15])


def run_experiment():
    import math
    model, cfg, itos, stoi, val_x, val_y, genes, device = get_baseline_and_data()

    print("Evaluating Baseline (original weights)...")
    base_nll, base_ppl = eval_nll(model, val_x, val_y)
    base_gen = eval_generation(model, device, stoi, itos, genes)
    base_loop, base_gen_str = eval_greedy_ctg_loop(model, device, stoi, itos)

    print(f"Baseline: NLL={base_nll:.4f}, PPL={base_ppl:.2f}")
    print(f"Generation: Stop rate={base_gen['stop_rate']*100:.1f}%, Hard caps={base_gen['hard_cap_rate']*100:.1f}%, Mean len={base_gen['mean_len']:.1f}, Mean GC={base_gen['mean_gc']*100:.1f}%")
    print(f"Greedy CTG loop: {base_loop} -> {base_gen_str}")
    print("=" * 60)

    # Let's test weight-space interventions!
    head_dim = model.n_embd // model.n_head # 48
    
    # We will test interventions:
    # 1. Dampen Layer 8 Head 1 and Layer 9 Head 1 output projection
    # 2. Add calibrated bias or weight adjustment for stop codons
    results = []

    for scale_l9 in [0.8, 0.5, 0.2, 0.0]:
        for scale_l8 in [1.0, 0.5, 0.0]:
            # Make a clean deepcopy of the state dict
            modified_state = copy.deepcopy(model.state_dict())
            
            # Edit Layer 9 Head 1 projection
            # blk.attn.proj.weight shape is [384, 384]
            # Head 1 corresponds to columns [48:96]
            h1_slice = slice(head_dim * 1, head_dim * 2)
            modified_state["blocks.9.attn.proj.weight"][:, h1_slice] *= scale_l9
            if scale_l8 < 1.0:
                modified_state["blocks.8.attn.proj.weight"][:, h1_slice] *= scale_l8

            # Load into test model
            test_model = build_codon_model_from_cfg(cfg)
            test_model.load_state_dict(modified_state)
            test_model.to(device)
            test_model.eval()

            nll, ppl = eval_nll(test_model, val_x, val_y)
            gen_res = eval_generation(test_model, device, stoi, itos, genes)
            is_loop, gen_str = eval_greedy_ctg_loop(test_model, device, stoi, itos)

            nll_diff = nll - base_nll
            print(f"Edit [L9_H1 scale={scale_l9}, L8_H1 scale={scale_l8}]:")
            print(f"  NLL={nll:.4f} (Δ={nll_diff:+.4f}), PPL={ppl:.2f}")
            print(f"  Stops={gen_res['stop_rate']*100:.1f}%, HardCaps={gen_res['hard_cap_rate']*100:.1f}%, MeanGC={gen_res['mean_gc']*100:.1f}%")
            print(f"  Loop: {is_loop} -> {gen_str[:40]}...")
            print("-" * 50)


if __name__ == "__main__":
    run_experiment()
