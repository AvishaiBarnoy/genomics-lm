import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
import json
import argparse
import yaml
from pathlib import Path
from src.protein_lm.tokenizer import ProteinTokenizer
from src.protein_lm.models_multi import MultiTaskProteinClassifier
from src.protein_lm.config import ProteinClassifierConfig

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

def train_multi_task(config_path, resume_path=None, run_id=None):
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
        num_classes=0, # Dummy value for multi-task backbone
        use_checkpoint=cfg.get("use_checkpoint", False)
    )

    print("[*] Building model...")
    model = MultiTaskProteinClassifier(model_cfg, task_dims).to(device)
    
    print("[*] Loading datasets...")
    train_ds = MultiTaskProteinDataset(cfg["train_data"], tokenizer, max_length=model_cfg.block_size)
    val_ds = MultiTaskProteinDataset(cfg["val_data"], tokenizer, max_length=model_cfg.block_size)
    
    train_loader = DataLoader(train_ds, batch_size=cfg.get("batch_size", 8), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.get("batch_size", 8))

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 1e-4)))
    
    # CrossEntropyLoss with ignore_index=-1 handles the missing labels
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    best_val_loss = float('inf')
    start_epoch = 0
    if resume_path and Path(resume_path).exists():
        print(f"[*] Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"[*] Resumed checkpoint. Next epoch: {start_epoch + 1} with best val loss: {best_val_loss:.4f}")

    print("[*] Starting Multi-Task Training...")
    epochs = cfg.get("epochs", 5)
    grad_accum_steps = cfg.get("grad_accum_steps", 1)
    print(f"[*] Gradient accumulation steps: {grad_accum_steps}")
    
    max_time_minutes = cfg.get("max_time_minutes", None)
    max_time_seconds = max_time_minutes * 60 if max_time_minutes else None
    if max_time_minutes:
        print(f"[*] Wall-time limit configured: {max_time_minutes} minutes")
    
    start_time = time.time()
    
    if not run_id:
        run_id = cfg.get("run_id", None)
    if not run_id:
        from datetime import date
        today = date.today().strftime("%Y-%m-%d")
        tag = Path(config_path).stem
        run_id = f"{today}_{tag}_{model_cfg.n_layer}L{model_cfg.n_head}H_d{model_cfg.n_embd}_e{epochs}"
    
    runs_dir = Path("runs") / run_id
    out_dir = runs_dir / "checkpoints"
    scores_dir = runs_dir / "scores"
    
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)
    
    log_csv = scores_dir / "curves.csv"
    import csv
    if not log_csv.exists():
        with open(log_csv, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_loss"])
    
    time_limit_reached = False
    for epoch in range(start_epoch, epochs):
        if time_limit_reached:
            break
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            
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
                loss = loss / grad_accum_steps
                loss.backward()
                train_loss += loss.item() * grad_accum_steps
            
            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
                
                if device.type == "mps":
                    torch.mps.empty_cache()
            elif (step + 1) % 250 == 0 and device.type == "mps":
                torch.mps.empty_cache()

            # Check wall-time limit at the end of every step
            if max_time_seconds and (time.time() - start_time) > max_time_seconds:
                print(f"\n[info] Wall-time limit of {max_time_minutes} minutes reached mid-epoch.")
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                }
                torch.save(checkpoint, out_dir / "last_critic.pt")
                print(f"[success] Gracefully saved checkpoint to {out_dir / 'last_critic.pt'}. Exiting.")
                time_limit_reached = True
                break
            
        train_loss /= len(train_loader)
        if device.type == "mps":
            torch.mps.empty_cache()
        
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
        if device.type == "mps":
            torch.mps.empty_cache()
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        with open(log_csv, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, f"{train_loss:.4f}", f"{val_loss:.4f}"])
        
        improved = False
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            improved = True

        # Save last checkpoint for resilience
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
        }
        torch.save(checkpoint, out_dir / "last_critic.pt")

        if improved:
            torch.save(model.state_dict(), out_dir / "best_critic.pt")
            print("  -> Saved new best model.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--resume", default=None, help="Path to checkpoint file to resume from")
    parser.add_argument("--run_id", default=None, help="Unique run id")
    args = parser.parse_args()
    train_multi_task(args.config, args.resume, args.run_id)
