#!/usr/bin/env python3
"""
Evaluate a sweep of epoch checkpoints (epoch_*.pt) on both perplexity 
and biological KPIs to find the best biological epoch, rename it to
best_biological.pt, and delete the suboptimal ones to save disk space.
"""

import argparse
import math
import json
import os
import csv
import re
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.codonlm.model_tiny_gpt import TinyGPT
from src.codonlm.eval_perplexity import PackedDataset
from scripts.sanity_kpis import compute_kpis, load_vocab, dev
from scripts._shared import resolve_run

def compute_perplexity(model, device, val_npz, batch_size=32):
    dataset = PackedDataset(val_npz)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    
    total_loss = 0.0
    total_tokens = 0
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction="sum")
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
            total_tokens += (y != 0).sum().item()
            
    avg_loss = total_loss / max(1, total_tokens)
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", help="Path to run checkpoints dir")
    ap.add_argument("--run_id", help="Run ID")
    ap.add_argument("--max_windows", type=int, default=200, help="Number of windows for sanity KPIs")
    ap.add_argument("--keep_all", action="store_true", help="Keep all epoch checkpoints (do not delete)")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    run_id, _ = resolve_run(args.run_id, args.run_dir)
    run_dir = Path(args.run_dir) if args.run_dir else (repo / "outputs" / "checkpoints" / run_id)
    scores_dir = repo / "outputs" / "scores" / run_id
    scores_dir.mkdir(parents=True, exist_ok=True)

    # 1. Find all epoch_*.pt files
    epoch_files = sorted(
        run_dir.glob("epoch_*.pt"),
        key=lambda p: int(re.search(r"epoch_(\d+)\.pt", p.name).group(1))
    )

    if not epoch_files:
        print(f"[!] No epoch_*.pt checkpoints found in {run_dir}.")
        print("[*] Please run training with --save_epochs flag first.")
        return

    print(f"[*] Found {len(epoch_files)} epoch checkpoints to evaluate.")

    # 2. Get dataset paths
    manifest_path = repo / "data/processed/combined" / run_id / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        val_npz = repo / manifest.get("val")
        test_npz = repo / manifest.get("test")
    else:
        # Defaults based on standard block size from first epoch's config
        first_state = torch.load(epoch_files[0], map_location="cpu")
        cfg = first_state.get("cfg", {})
        block_size = cfg.get("block_size", 256)
        val_npz = repo / f"data/processed/val_bs{block_size}.npz"
        test_npz = repo / f"data/processed/test_bs{block_size}.npz"

    itos, _ = load_vocab(run_id, repo)
    device = dev()
    
    sweep_results = []

    # 3. Evaluate each epoch
    for ep_file in epoch_files:
        epoch_idx = int(re.search(r"epoch_(\d+)\.pt", ep_file.name).group(1))
        print(f"\n[*] Evaluating Epoch {epoch_idx} ({ep_file.name})...")
        
        state = torch.load(ep_file, map_location="cpu")
        sd = state.get("model", state)
        cfg = state.get("cfg", {})
        
        # Build model
        model = TinyGPT(
            vocab_size=cfg["vocab_size"],
            block_size=cfg["block_size"],
            n_layer=cfg["n_layer"],
            n_head=cfg["n_head"],
            n_embd=cfg["n_embd"],
            dropout=0.0,
            use_sdpa=cfg.get("use_sdpa", False)
        )
        model.load_state_dict(sd, strict=False)
        model.to(device).eval()
        
        # Evaluate Perplexity
        val_loss, ppl = compute_perplexity(model, device, val_npz)
        
        # Evaluate biological KPIs
        kpis = compute_kpis(model, device, test_npz, itos, sample_windows=args.max_windows)
        
        result = {
            "epoch": epoch_idx,
            "val_loss": val_loss,
            "perplexity": ppl,
            "codon_corr": kpis["codon_corr"],
            "frameshift_delta": kpis["frameshift_delta"],
            "startstop_delta.start": kpis["startstop_delta.start"],
            "startstop_delta.stop": kpis["startstop_delta.stop"],
            "syn_gap": kpis["syn_gap"],
            "file_path": str(ep_file)
        }
        sweep_results.append(result)
        
        print(f"  Val Loss: {val_loss:.4f} | Perplexity: {ppl:.2f}")
        print(f"  Codon Corr: {kpis['codon_corr']:.4f} | Syn Gap: {kpis['syn_gap']:.4f} | Frameshift: {kpis['frameshift_delta']:.4f}")

    # 4. Multi-criteria ranking to find best biological epoch
    # We rank epochs on each biophysical metric: codon_corr, syn_gap, frameshift_delta, startstop_delta.start, startstop_delta.stop
    # Epoch with the lowest sum of ranks wins (Borda rank aggregation)
    metrics_to_rank = ["codon_corr", "syn_gap", "frameshift_delta", "startstop_delta.start", "startstop_delta.stop"]
    
    # Initialize rank sums
    rank_sums = {r["epoch"]: 0 for r in sweep_results}
    
    for metric in metrics_to_rank:
        # Sort results from best to worst (descending order for these positive metrics)
        sorted_eps = sorted(sweep_results, key=lambda x: x[metric], reverse=True)
        for rank, res in enumerate(sorted_eps):
            rank_sums[res["epoch"]] += rank
            
    best_epoch_idx = min(rank_sums, key=rank_sums.get)
    best_res = next(r for r in sweep_results if r["epoch"] == best_epoch_idx)
    
    print("\n" + "="*50)
    print(f"BEST BIOLOGICAL CHECKPOINT: Epoch {best_epoch_idx}")
    print(f"  Codon Correlation: {best_res['codon_corr']:.4f}")
    print(f"  Synonymous Gap: {best_res['syn_gap']:.4f}")
    print(f"  Frameshift Delta: {best_res['frameshift_delta']:.4f}")
    print(f"  Start Mutation Delta: {best_res['startstop_delta.start']:.4f}")
    print(f"  Stop Mutation Delta: {best_res['startstop_delta.stop']:.4f}")
    print(f"  Validation Perplexity: {best_res['perplexity']:.2f}")
    print("="*50)

    # Save results to CSV
    csv_path = scores_dir / "epoch_sweep_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "val_loss", "perplexity", "codon_corr", "frameshift_delta", "startstop_delta.start", "startstop_delta.stop", "syn_gap", "rank_sum"])
        for res in sweep_results:
            w.writerow([
                res["epoch"],
                f"{res['val_loss']:.4f}",
                f"{res['perplexity']:.4f}",
                f"{res['codon_corr']:.4f}",
                f"{res['frameshift_delta']:.4f}",
                f"{res['startstop_delta.start']:.4f}",
                f"{res['startstop_delta.stop']:.4f}",
                f"{res['syn_gap']:.4f}",
                rank_sums[res["epoch"]]
            ])
    print(f"\n[save] Detailed sweep results written to {csv_path}")

    # 5. Save best biological checkpoint and clean up
    best_biological_path = run_dir / "best_biological.pt"
    # Load and save the winning checkpoint as best_biological.pt
    import shutil
    shutil.copy(best_res["file_path"], best_biological_path)
    print(f"[save] Copied best biological checkpoint to {best_biological_path}")

    # Delete other epoch checkpoints if not keeping them
    if not args.keep_all:
        print("\n[*] Cleaning up suboptimal checkpoints to save space...")
        deleted_count = 0
        for res in sweep_results:
            p = Path(res["file_path"])
            if p.exists() and res["epoch"] != best_epoch_idx:
                p.unlink()
                deleted_count += 1
        print(f"[cleanup] Deleted {deleted_count} epoch checkpoints. Kept epoch_{best_epoch_idx}.pt (as best_biological.pt).")
    else:
        print("\n[*] --keep_all active. No files deleted.")

if __name__ == "__main__":
    main()
