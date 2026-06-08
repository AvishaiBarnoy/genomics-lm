import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
import argparse
import yaml
import math
from pathlib import Path
from src.protein_lm.tokenizer import ProteinTokenizer
from src.protein_lm.models_multi import MultiTaskProteinClassifier
from src.protein_lm.config import ProteinClassifierConfig, load_config
from scripts._shared import ensure_run_layout, write_meta

class MultiTaskProteinDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        with open(jsonl_path, "r") as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        
        # Tokenize (Add BOS, pad/truncate, Add EOS handled by tokenizer logic if needed, but we keep it simple here)
        # ProteinTokenizer currently handles sequence encoding directly.
        # We manually add BOS (1) and EOS (2) if they exist, or just rely on raw tokens.
        tokens = [self.tokenizer.bos_token_id] + self.tokenizer.encode_sequence(s["sequence"])[:self.max_length-2] + [self.tokenizer.eos_token_id]
        
        # Pad sequence
        pad_len = self.max_length - len(tokens)
        input_ids = tokens + [self.tokenizer.pad_token_id] * pad_len
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "family": torch.tensor(s.get("pfam_id", -1), dtype=torch.long),
            "function": torch.tensor(s.get("ec_id", -1), dtype=torch.long),
            "stability": torch.tensor(s.get("stability_id", -1), dtype=torch.long)
        }

def train_multi_task(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
        
    device_name = cfg.get("device", "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(device_name)
    print(f"[*] Using device: {device}")

    tokenizer = ProteinTokenizer()
    
    # Load Vocabs to get task dimensions
    with open("data/processed/protein_lm/multitask/task_vocabs.json", "r") as f:
        vocabs = json.load(f)
        
    task_dims = {
        "family": len(vocabs["pfam"]),
        "function": len(vocabs["ec"]),
        "stability": len(vocabs["stability"])
    }
    print(f"[*] Task Dimensions: {task_dims}")

    # Build Config
    model_cfg = ProteinClassifierConfig(
        vocab_size=len(tokenizer.vocab),
        block_size=cfg.get("block_size", 512),
        n_layer=cfg.get("n_layer", 4),
        n_head=cfg.get("n_head", 4),
        n_embd=cfg.get("n_embd", 128),
        dropout=cfg.get("dropout", 0.1),
        num_classes=0 # Dummy value for multi-task backbone
    )

    print(f"[*] Building model...")
    model = MultiTaskProteinClassifier(model_cfg, task_dims).to(device)
    
    print(f"[*] Loading datasets...")
    train_ds = MultiTaskProteinDataset(cfg["train_data"], tokenizer, max_length=model_cfg.block_size)
    val_ds = MultiTaskProteinDataset(cfg["val_data"], tokenizer, max_length=model_cfg.block_size)
    
    train_loader = DataLoader(train_ds, batch_size=cfg.get("batch_size", 8), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.get("batch_size", 8))

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 1e-4)))
    
    # CrossEntropyLoss with ignore_index=-1 handles the missing labels
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    print(f"[*] Starting Multi-Task Training...")
    
    best_val_loss = float('inf')
    epochs = cfg.get("epochs", 5)
    
    out_dir = Path(cfg.get("out_dir", "outputs/checkpoints/protein_critic"))
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            
            optimizer.zero_grad()
            logits_dict = model(input_ids)
            
            loss = 0
            tasks_added = 0
            for task in ["family", "function", "stability"]:
                targets = batch[task].to(device)
                # Check if there's at least one valid label in the batch for this task
                if (targets != -1).any():
                    loss += criterion(logits_dict[task], targets)
                    tasks_added += 1
            
            if tasks_added > 0:
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            else:
                optimizer.step() # Still step optimizer to avoid potential stale grads? Or just skip.
            
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0.0
        val_tasks_total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                logits_dict = model(input_ids)
                
                batch_loss = 0
                batch_tasks = 0
                for task in ["family", "function", "stability"]:
                    targets = batch[task].to(device)
                    if (targets != -1).any():
                        batch_loss += criterion(logits_dict[task], targets)
                        batch_tasks += 1
                
                if batch_tasks > 0:
                    val_loss += batch_loss.item()
                    val_tasks_total += 1
        
        if val_tasks_total > 0:
            val_loss /= val_tasks_total
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), out_dir / "best_critic.pt")
            print(f"  -> Saved new best model.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()
    train_multi_task(args.config)
