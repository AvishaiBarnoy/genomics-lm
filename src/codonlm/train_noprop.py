#!/usr/bin/env python3
"""
Training script for NoPropTinyGPT using local layer-wise MSE denoising targets
instead of global backpropagation.
"""

import argparse
import yaml
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from .model_tiny_gpt import NoPropTinyGPT
from .train_codon_lm import PackedDataset, _ensure_path_list, _normalize_run_id, _auto_run_id, _prepare_output_dirs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run_id", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--noise_sigma", type=float, default=0.1, help="Sigma of Gaussian noise added to targets")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"[noprop] using device: {device}")

    # Set run ID
    run_id = _normalize_run_id(args.run_id) or _auto_run_id(cfg, args.config)
    ckpt_root, scores_root = _prepare_output_dirs("runs", "runs", run_id)
    print(f"[noprop] run_id: {run_id}")

    # Load datasets
    train_paths = _ensure_path_list(None, cfg.get("train_npz"), "train_npz")
    val_paths = _ensure_path_list(None, cfg.get("val_npz"), "val_npz")
    train_ds = PackedDataset(train_paths)
    val_ds = PackedDataset(val_paths)

    collate_fn = None
    if getattr(train_ds, "is_dynamic", False):
        def dynamic_collate_fn(batch):
            lengths = [len(seq) for seq in batch]
            max_len = max(lengths)
            xs, ys = [], []
            for seq in batch:
                x_seq = seq[:-1]
                y_seq = seq[1:]
                pad_len = (max_len - 1) - len(x_seq)
                if pad_len > 0:
                    x_seq = torch.cat([x_seq, torch.zeros(pad_len, dtype=torch.long)])
                    y_seq = torch.cat([y_seq, torch.zeros(pad_len, dtype=torch.long)])
                xs.append(x_seq)
                ys.append(y_seq)
            return torch.stack(xs), torch.stack(ys)
        collate_fn = dynamic_collate_fn

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], collate_fn=collate_fn)

    # Initialize model
    model = NoPropTinyGPT(
        vocab_size=cfg["vocab_size"],
        block_size=cfg["block_size"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        n_embd=cfg["n_embd"],
        dropout=cfg.get("dropout", 0.1),
        sep_id=(3 if cfg.get("sep_mask_enabled", True) else None),
        n_kv_head=cfg.get("n_kv_head", None),
        use_sdpa=cfg.get("use_sdpa", False)
    ).to(device)

    # Initialize layer-wise optimizers
    lr = float(cfg.get("learning_rate", 5e-4))

    # We create separate optimizers for:
    # 1. Embedding layer
    # 2. Each NoProp block
    # 3. Output layer (LN_F and LM head)
    opt_emb = torch.optim.AdamW(list(model.tok_emb.parameters()) + list(model.pos_emb.parameters()), lr=lr)
    opts_blocks = [torch.optim.AdamW(block.parameters(), lr=lr) for block in model.blocks]
    opt_head = torch.optim.AdamW(list(model.ln_f.parameters()) + list(model.head.parameters()), lr=lr)

    epochs = cfg.get("epochs", 5)
    noise_sigma = args.noise_sigma

    print(f"[noprop] starting training for {epochs} epochs")
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # Training loop
        model.train()
        train_loss_ce = 0.0
        train_loss_blocks = [0.0] * len(model.blocks)
        n_train = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            B, T = x.shape
            n_train += 1

            # Get target embeddings
            y_clean = model.tok_emb(y).detach()  # (B, T, C)
            noise = torch.randn_like(y_clean) * noise_sigma
            y_noisy = y_clean + noise
            non_pad_mask = (y != 0).unsqueeze(-1).float()

            # 1. Step embeddings
            opt_emb.zero_grad()
            pos = torch.arange(0, T, device=x.device).unsqueeze(0)
            h = model.tok_emb(x) + model.pos_emb(pos)
            h = model.drop(h)

            attn_mask = None
            if model.sep_id is not None:
                sep = (x == int(model.sep_id))
                seg = torch.cumsum(sep, dim=1)
                attn_mask = (seg.unsqueeze(-1) == seg.unsqueeze(-2)).unsqueeze(1)

            # Keep track of outputs without backpropagating between layers
            h_prev = h

            # Step-by-step block-wise training
            for l, block in enumerate(model.blocks):
                # Detach context input from graph to stop gradient propagation
                h_in = h_prev.detach() if l > 0 else h_prev

                opts_blocks[l].zero_grad()
                h_out, pred_y = block(h_in, noisy_targets=y_noisy, attn_mask=attn_mask)

                loss_mse_elementwise = F.mse_loss(pred_y, y_clean, reduction="none")
                loss_mse = (loss_mse_elementwise * non_pad_mask).sum() / (non_pad_mask.sum() * pred_y.size(-1) + 1e-8)
                loss_mse.backward()

                opts_blocks[l].step()
                if l == 0:
                    # Also update embeddings via block 0 output
                    opt_emb.step()

                train_loss_blocks[l] += loss_mse.item()
                h_prev = h_out

            # 2. Step final head
            opt_head.zero_grad()
            h_final = model.ln_f(h_prev.detach())
            logits = model.head(h_final)

            loss_ce = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=0)
            loss_ce.backward()
            opt_head.step()

            train_loss_ce += loss_ce.item()

        # Validation loop
        model.eval()
        val_loss_ce = 0.0
        val_loss_blocks = [0.0] * len(model.blocks)
        n_val = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                B, T = x.shape
                n_val += 1

                y_clean = model.tok_emb(y)
                non_pad_mask = (y != 0).unsqueeze(-1).float()
                pos = torch.arange(0, T, device=x.device).unsqueeze(0)
                h = model.tok_emb(x) + model.pos_emb(pos)
                h = model.drop(h)

                attn_mask = None
                if model.sep_id is not None:
                    sep = (x == int(model.sep_id))
                    seg = torch.cumsum(sep, dim=1)
                    attn_mask = (seg.unsqueeze(-1) == seg.unsqueeze(-2)).unsqueeze(1)

                h_prev = h
                for l, block in enumerate(model.blocks):
                    h_out, pred_y = block(h_prev, noisy_targets=y_clean, attn_mask=attn_mask)
                    loss_mse_elementwise = F.mse_loss(pred_y, y_clean, reduction="none")
                    loss_mse = (loss_mse_elementwise * non_pad_mask).sum() / (non_pad_mask.sum() * pred_y.size(-1) + 1e-8)
                    val_loss_blocks[l] += loss_mse.item()
                    h_prev = h_out

                h_final = model.ln_f(h_prev)
                logits = model.head(h_final)
                loss_ce = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=0)
                val_loss_ce += loss_ce.item()

        # Print statistics
        print(f"Epoch {epoch}/{epochs}:")
        train_block_str = ", ".join([f"B{i}:{train_loss_blocks[i]/n_train:.4f}" for i in range(len(model.blocks))])
        val_block_str = ", ".join([f"B{i}:{val_loss_blocks[i]/n_val:.4f}" for i in range(len(model.blocks))])
        print(f"  Train Block MSEs: {train_block_str} | CE Loss: {train_loss_ce/n_train:.4f}")
        print(f"  Val Block MSEs:   {val_block_str} | CE Loss: {val_loss_ce/n_val:.4f}")

        # Save checkpoint
        avg_val_loss = val_loss_ce / n_val
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = ckpt_root / "best.pt"
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "val_loss": avg_val_loss
            }, ckpt_path)
            print(f"  [noprop] saved new best checkpoint to {ckpt_path}")

if __name__ == "__main__":
    main()
