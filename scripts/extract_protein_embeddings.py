#!/usr/bin/env python3
"""
Stage 3/4: Protein Embedding Extractor (NoProp / Frozen Backbone)
Extracts mean-pooled sequence representations from a pre-trained Protein Critic backbone
and saves them as static NPZ feature files.
"""

import argparse
import json
import yaml
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.protein_lm.tokenizer import ProteinTokenizer
from src.protein_lm.models_multi import MultiTaskProteinClassifier
from src.protein_lm.config import ProteinClassifierConfig
from src.protein_lm.train_multi_task import MultiTaskProteinDataset

def dev() -> torch.device:
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

def extract_embeddings(config_path, ckpt_path, vocabs_path, data_path, out_npz):
    device = dev()
    print(f"[*] Using device: {device}")

    # Load vocabs for task dimensions
    with open(vocabs_path, "r") as f:
        vocabs = json.load(f)
    task_dims = {
        "family": len(vocabs["pfam"]),
        "function": len(vocabs["ec"]),
        "stability": len(vocabs["stability"])
    }

    # Load configuration
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    tokenizer = ProteinTokenizer()
    model_cfg = ProteinClassifierConfig(
        vocab_size=len(tokenizer.vocab),
        block_size=cfg.get("block_size", 512),
        n_layer=cfg.get("n_layer", 4),
        n_head=cfg.get("n_head", 4),
        n_embd=cfg.get("n_embd", 128),
        dropout=0.0,
        num_classes=0
    )

    print("[*] Rebuilding model...")
    model = MultiTaskProteinClassifier(model_cfg, task_dims)
    
    # Load weights if checkpoint exists
    if Path(ckpt_path).exists():
        print(f"[*] Loading weights from {ckpt_path}...")
        state = torch.load(ckpt_path, map_location="cpu")
        # Filter state dict to only load the backbone parameters
        backbone_state = {k: v for k, v in state.items() if k.startswith("backbone.")}
        missing, unexpected = model.load_state_dict(backbone_state, strict=False)
        print(f"[*] Successfully loaded backbone. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    else:
        print(f"[!] Warning: Checkpoint {ckpt_path} not found. Extracting with initialized weights.")
        
    model.to(device)
    model.eval()

    # Load Dataset
    print(f"[*] Loading dataset from {data_path}...")
    dataset = MultiTaskProteinDataset(data_path, tokenizer, max_length=model_cfg.block_size)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    all_feats = []
    all_families = []
    all_functions = []
    all_stabilities = []

    print("[*] Extracting sequence embeddings...")
    with torch.no_grad():
        for batch in tqdm(loader):
            input_ids = batch["input_ids"].to(device)
            
            # Forward pass through backbone blocks
            seq_length = input_ids.size(1)
            token_embeds = model.backbone.token_embedding(input_ids)
            pos_embeds = model.backbone.position_embedding(torch.arange(seq_length, device=input_ids.device))
            x = model.backbone.dropout(token_embeds + pos_embeds)
            
            for block in model.backbone.transformer_blocks:
                x = block(x)
                
            # Mean-pool over sequence length
            pooled = x.mean(dim=1)
            
            all_feats.append(pooled.cpu().numpy())
            all_families.append(batch["family"].numpy())
            all_functions.append(batch["function"].numpy())
            all_stabilities.append(batch["stability"].numpy())

    X = np.concatenate(all_feats, axis=0)
    y_family = np.concatenate(all_families, axis=0)
    y_function = np.concatenate(all_functions, axis=0)
    y_stability = np.concatenate(all_stabilities, axis=0)

    # Save as NPZ
    out_path = Path(out_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        X=X,
        y_family=y_family,
        y_function=y_function,
        y_stability=y_stability
    )
    print(f"[success] Saved embeddings of shape {X.shape} and labels to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/protein_critic.yaml", help="Path to YAML config")
    parser.add_argument("--ckpt", default="runs/protein_critic/checkpoints/best_critic.pt", help="Path to model weights")
    parser.add_argument("--vocabs", default="data/processed/protein_lm/multitask/task_vocabs.json", help="Path to task vocabs json")
    parser.add_argument("--data", required=True, help="Path to input JSONL dataset")
    parser.add_argument("--out", required=True, help="Path to output NPZ file")
    args = parser.parse_args()
    
    extract_embeddings(args.config, args.ckpt, args.vocabs, args.data, args.out)
