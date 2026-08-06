#!/usr/bin/env python3
"""
Train Protein Latent Energy-Based Model (EBM) via Noise Contrastive Estimation (NCE).
"""

from __future__ import annotations

import argparse
import random
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path

from src.protein_lm.tokenizer import ProteinTokenizer
from src.protein_lm.dataset import MultiTaskProteinDataset, LengthBucketBatchSampler, collate_protein_batch
from src.protein_lm.models_multi import MultiTaskProteinClassifier
from src.protein_lm.config import ProteinClassifierConfig
from src.protein_lm.ebm import ProteinLatentEBM
from src.training.run_lifecycle import (
    TrainingRun,
    capture_rng_state,
    configuration_fingerprint,
    restore_rng_state,
)
from src.training.runtime import save_checkpoint_atomic

AMINO_ACIDS = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

def corrupt_sequence(seq: str, mutation_rate: float = 0.20) -> str:
    """Generate negative noise sequence by substituting residues randomly."""
    seq_list = list(seq)
    n_mutations = max(1, int(len(seq) * mutation_rate))
    indices = random.sample(range(len(seq)), n_mutations)
    for idx in indices:
        seq_list[idx] = random.choice(AMINO_ACIDS)
    return "".join(seq_list)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Protein Latent EBM.")
    parser.add_argument("--config", required=True, help="Path to critic config file")
    parser.add_argument("--critic_ckpt", required=True, help="Path to pre-trained MultiTask critic checkpoint")
    parser.add_argument("--epochs", type=int, default=5, help="Number of EBM training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for EBM")
    parser.add_argument("--pooling", default="attention", help="Backbone pooling type (attention | mean)")
    parser.add_argument("--family_dim", type=int, default=2000, help="Classifier family dimension")
    parser.add_argument("--function_dim", type=int, default=1000, help="Classifier function dimension")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden size of the EBM network")
    parser.add_argument("--out_dir", default="runs/protein_ebm", help="Output directory for checkpoints")
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--seed", type=int, default=1337, help="RNG seed")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"[ebm-train] using device: {device}")

    # Load configuration
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    requested_dir = Path(args.out_dir)
    run_id = args.run_id or requested_dir.name
    fingerprint = configuration_fingerprint(
        {
            **cfg,
            "critic_ckpt": str(Path(args.critic_ckpt).resolve()),
            "lr": args.lr,
            "pooling": args.pooling,
            "family_dim": args.family_dim,
            "function_dim": args.function_dim,
            "hidden_dim": args.hidden_dim,
            "seed": args.seed,
        }
    )
    training_run = TrainingRun.open(
        requested_dir.parent,
        run_id,
        resume=args.resume,
        last_checkpoint_name="last_ebm.pt",
        target_epochs=args.epochs,
        config_fingerprint=fingerprint,
    )
    run_logger = training_run.logger()
    run_logger.__enter__()

    # Initialize tokenizer and datasets
    tokenizer = ProteinTokenizer()
    print("[ebm-train] loading dataset paths from config")
    train_dataset = MultiTaskProteinDataset(cfg["train_data"], tokenizer, max_length=cfg.get("block_size", 512))
    val_dataset = MultiTaskProteinDataset(cfg["val_data"], tokenizer, max_length=cfg.get("block_size", 512))

    batch_size = cfg.get("batch_size", 4)
    train_sampler = LengthBucketBatchSampler(train_dataset, batch_size=batch_size, shuffle=True)
    val_sampler = LengthBucketBatchSampler(val_dataset, batch_size=batch_size, shuffle=False)

    def collate_fn(batch):
        return collate_protein_batch(batch, pad_token_id=tokenizer.pad_token_id)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, collate_fn=collate_fn)

    # Initialize frozen MultiTask backbone classifier
    model_cfg = ProteinClassifierConfig(
        vocab_size=len(tokenizer),
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        n_embd=cfg["n_embd"],
        block_size=cfg["block_size"],
        dropout=cfg["dropout"],
        pooling=args.pooling,
        num_classes=2,
    )
    
    # Standard task dimensions used by MultiTask critic
    task_dims = {"family": args.family_dim, "function": args.function_dim, "stability": 2}
    print(f"[ebm-train] loading MultiTask critic from {args.critic_ckpt}")
    critic = MultiTaskProteinClassifier(model_cfg, task_dims)
    
    state_dict = torch.load(args.critic_ckpt, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]
    critic.load_state_dict(state_dict)
    critic.to(device)
    
    # Freeze critic parameters completely
    for p in critic.parameters():
        p.requires_grad = False
    critic.eval()

    # Initialize Energy-Based Model head
    ebm = ProteinLatentEBM(n_embd=cfg["n_embd"], hidden_dim=args.hidden_dim)
    ebm.to(device)

    optimizer = torch.optim.AdamW(ebm.parameters(), lr=args.lr, weight_decay=0.01)

    out_dir = training_run.run_dir
    ckpt_dir = training_run.checkpoints
    scores_dir = training_run.scores

    # Copy config
    import shutil
    try:
        shutil.copy(args.config, ckpt_dir / "config.yaml")
    except Exception as e:
        print(f"[ebm-train] warning: failed to copy config file: {e}")

    # Initialize curves.csv
    curves_path = scores_dir / "curves.csv"
    if not curves_path.exists():
        with open(curves_path, "w") as f:
            f.write("epoch,train_loss,val_loss\n")

    print(f"[ebm-train] starting EBM training: epochs={args.epochs}, lr={args.lr}")
    
    best_val_loss = float("inf")
    best_epoch = 0
    start_epoch = 1
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        ebm.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        start_epoch = int(checkpoint["epoch"]) + 1
        restore_rng_state(checkpoint.get("rng_state"))

    for epoch in range(start_epoch, args.epochs + 1):
        ebm.train()
        total_loss = 0.0
        n_batches = 0

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()

            # 1. Prepare positive and corrupted negative sequences
            pos_seqs = batch["sequence"]
            neg_seqs = [corrupt_sequence(seq, mutation_rate=0.20) for seq in pos_seqs]

            # Tokenize negative batch
            neg_batch_ids = []
            neg_batch_mask = []
            for seq in neg_seqs:
                tokens = [tokenizer.bos_token_id] + tokenizer.encode_sequence(seq)[:cfg["block_size"] - 2] + [tokenizer.eos_token_id]
                pad_len = batch["input_ids"].shape[1] - len(tokens)
                if pad_len > 0:
                    neg_batch_ids.append(tokens + [tokenizer.pad_token_id] * pad_len)
                    neg_batch_mask.append([1] * len(tokens) + [0] * pad_len)
                else:
                    neg_batch_ids.append(tokens[:batch["input_ids"].shape[1]])
                    neg_batch_mask.append([1] * batch["input_ids"].shape[1])

            pos_ids = batch["input_ids"].to(device)
            pos_mask = batch["attention_mask"].to(device)
            neg_ids = torch.tensor(neg_batch_ids, dtype=torch.long, device=device)
            neg_mask = torch.tensor(neg_batch_mask, dtype=torch.long, device=device)

            # 2. Extract continuous latent representations from frozen critic backbone
            with torch.no_grad():
                z_pos = critic.extract_latent(pos_ids, pos_mask)
                z_neg = critic.extract_latent(neg_ids, neg_mask)

            # 3. Calculate energies
            energy_pos = ebm(z_pos)
            energy_neg = ebm(z_neg)

            # 4. softplus ranking loss (minimize energy of real, maximize energy of mutated decoys)
            loss = torch.mean(F.softplus(energy_pos - energy_neg))
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            if step % 50 == 0:
                print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f} | E_pos: {energy_pos.mean().item():.3f} | E_neg: {energy_neg.mean().item():.3f}")

        # Validation phase
        ebm.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                pos_seqs = batch["sequence"]
                neg_seqs = [corrupt_sequence(seq, mutation_rate=0.20) for seq in pos_seqs]

                neg_batch_ids = []
                neg_batch_mask = []
                for seq in neg_seqs:
                    tokens = [tokenizer.bos_token_id] + tokenizer.encode_sequence(seq)[:cfg["block_size"] - 2] + [tokenizer.eos_token_id]
                    pad_len = batch["input_ids"].shape[1] - len(tokens)
                    if pad_len > 0:
                        neg_batch_ids.append(tokens + [tokenizer.pad_token_id] * pad_len)
                        neg_batch_mask.append([1] * len(tokens) + [0] * pad_len)
                    else:
                        neg_batch_ids.append(tokens[:batch["input_ids"].shape[1]])
                        neg_batch_mask.append([1] * batch["input_ids"].shape[1])

                pos_ids = batch["input_ids"].to(device)
                pos_mask = batch["attention_mask"].to(device)
                neg_ids = torch.tensor(neg_batch_ids, dtype=torch.long, device=device)
                neg_mask = torch.tensor(neg_batch_mask, dtype=torch.long, device=device)

                z_pos = critic.extract_latent(pos_ids, pos_mask)
                z_neg = critic.extract_latent(neg_ids, neg_mask)

                energy_pos = ebm(z_pos)
                energy_neg = ebm(z_neg)
                val_loss += torch.mean(F.softplus(energy_pos - energy_neg)).item()
                val_batches += 1

        avg_train = total_loss / n_batches
        avg_val = val_loss / val_batches
        print(f"--- Epoch {epoch} Complete | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} ---")

        # Append to curves.csv
        with open(curves_path, "a") as f:
            f.write(f"{epoch},{avg_train:.6f},{avg_val:.6f}\n")

        # Save checkpoint
        payload = {
            "model": ebm.state_dict(),
            "epoch": epoch,
            "val_loss": avg_val,
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": min(best_val_loss, avg_val),
            "best_epoch": epoch if avg_val < best_val_loss else best_epoch,
            "epoch_complete": True,
            "run_fingerprint": fingerprint,
            "rng_state": capture_rng_state(),
            "run_progress": {
                "completed_epochs": epoch,
                "current_epoch": epoch,
                "microbatch": 0,
                "optimizer_step": epoch * len(train_loader),
            },
        }
        
        # Save epoch checkpoint
        ckpt_path = ckpt_dir / f"ebm_epoch_{epoch}.pt"
        save_checkpoint_atomic(payload, ckpt_path)
        print(f"[saved] {ckpt_path}")

        # Save last checkpoint
        last_path = ckpt_dir / "last_ebm.pt"
        save_checkpoint_atomic(payload, last_path)
        
        # Save best checkpoint
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_epoch = epoch
            best_path = ckpt_dir / "best_ebm.pt"
            save_checkpoint_atomic(payload, best_path)
            save_checkpoint_atomic(payload, ckpt_dir / f"best_ebm_epoch_{epoch:03d}.pt")
            print(f"[saved] {best_path} (new best validation loss: {best_val_loss:.4f})")

    # Generate summary.md
    summary_path = out_dir / "summary.md"
    summary_content = f"""# Run Summary: `{out_dir.name}`

## 📊 Status & Key Performance Indicators (KPIs)
- **Status:** Completed
- **Epochs Trained:** {args.epochs}
- **Best Epoch:** {best_epoch}
- **Best Validation Loss:** {best_val_loss:.4f}

## 🧠 Model Architecture & Settings
- **Type:** Protein Latent Energy-Based Model (EBM)
- **Latent Dim:** {cfg["n_embd"]}
- **EBM Hidden Dim:** {args.hidden_dim}
- **Pooling:** {args.pooling}
- **Backbone Critic Checkpoint:** `{args.critic_ckpt}`
"""
    summary_path.write_text(summary_content)
    training_run.mark_complete(
        {"completed_epochs": args.epochs, "best_epoch": best_epoch,
         "best_validation_loss": best_val_loss}
    )
    training_run.close()
    print(f"[summary] Wrote run summary to {summary_path}")
    print("[ebm-train] Done training!")


if __name__ == "__main__":
    main()
