import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import argparse
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score

class TaskFeatureDataset(Dataset):
    def __init__(self, npz_path, task_name):
        data = np.load(npz_path)
        X = torch.tensor(data["X"], dtype=torch.float32)
        y = torch.tensor(data[f"y_{task_name}"], dtype=torch.long)
        
        # Filter out unlabelled samples (where label is -1)
        mask = y != -1
        self.X = X[mask]
        self.y = y[mask]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            "X": self.X[idx],
            "label": self.y[idx]
        }

class MultiTaskMLPClassifier(nn.Module):
    def __init__(self, input_dim, task_dims):
        super().__init__()
        # 2-layer MLP head per task
        self.heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, dim)
            ) for name, dim in task_dims.items()
        })

    def forward(self, x):
        return {name: head(x) for name, head in self.heads.items()}

def train_mlp_heads(train_npz, val_npz, vocabs_json, epochs=100, batch_size=64, lr=1e-3, out_dir="runs/protein_critic/checkpoints"):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[*] Using device: {device}")

    # Load vocabs
    with open(vocabs_json, "r") as f:
        vocabs = json.load(f)
    task_dims = {
        "family": len(vocabs["pfam"]),
        "function": len(vocabs["ec"]),
        "stability": len(vocabs["stability"])
    }

    # Initialize model
    # Load first npz to inspect dimensions
    dummy_data = np.load(train_npz)
    input_dim = dummy_data["X"].shape[1]
    
    model = MultiTaskMLPClassifier(input_dim, task_dims).to(device)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"[*] Training independent MLP heads on {input_dim}-dim features...")

    for task_name, task_dim in task_dims.items():
        print(f"\n[*] Task: {task_name} (dimension: {task_dim})")
        train_ds = TaskFeatureDataset(train_npz, task_name)
        val_ds = TaskFeatureDataset(val_npz, task_name)
        
        if len(train_ds) == 0:
            print(f"[!] Warning: No training samples for task {task_name}. Skipping.")
            continue
            
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        head = model.heads[task_name]
        optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        best_head_state = None

        print(f"  Training on {len(train_ds)} samples, validating on {len(val_ds)} samples...")

        for epoch in range(epochs):
            head.train()
            train_loss = 0.0
            for batch in train_loader:
                X = batch["X"].to(device)
                targets = batch["label"].to(device)

                optimizer.zero_grad()
                logits = head(X)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            head.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    X = batch["X"].to(device)
                    targets = batch["label"].to(device)
                    logits = head(X)
                    loss = criterion(logits, targets)
                    val_loss += loss.item()
            val_loss /= len(val_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_head_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:03d}/{epochs:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Load best state for this head
        head.load_state_dict(best_head_state)
        print(f"[success] Completed training {task_name} head. Best Val Loss: {best_val_loss:.4f}")

    # Save overall model state dict
    torch.save(model.state_dict(), out_path / "mlp_heads.pt")
    print(f"\n[success] Saved unified MLP heads checkpoint to {out_path / 'mlp_heads.pt'}")

    # Final evaluation with Top-K metrics
    print("\n[*] Evaluating final independent MLP heads...")
    model.eval()

    results = {task: {"preds": [], "targets": [], "top5": [], "top10": []} for task in task_dims}

    with torch.no_grad():
        for task in task_dims:
            val_ds = TaskFeatureDataset(val_npz, task)
            if len(val_ds) == 0:
                continue
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
            head = model.heads[task]
            head.eval()
            
            for batch in val_loader:
                X = batch["X"].to(device)
                targets = batch["label"]
                logits = head(X)

                preds = torch.argmax(logits, dim=-1)
                results[task]["preds"].extend(preds.cpu().tolist())
                results[task]["targets"].extend(targets.tolist())

                # Top-K
                num_classes = logits.size(-1)
                k_val = min(10, num_classes)
                _, topk_idx = torch.topk(logits, k=k_val, dim=-1)
                correct_topk = (topk_idx == targets.unsqueeze(-1).to(device))

                k5 = min(5, num_classes)
                has_top5 = correct_topk[:, :k5].any(dim=-1).cpu().tolist()
                has_top10 = correct_topk[:, :k_val].any(dim=-1).cpu().tolist()

                results[task]["top5"].extend(has_top5)
                results[task]["top10"].extend(has_top10)

    for task in task_dims:
        y_true = results[task]["targets"]
        y_pred = results[task]["preds"]
        if len(y_true) > 0:
            acc = accuracy_score(y_true, y_pred)
            print("==================================================")
            print(f"Task: {task}")
            print(f"  Samples evaluated: {len(y_true)}")
            print(f"  Top-1 Accuracy: {acc:.4f}")
            if task in ["family", "function"]:
                top5_acc = sum(results[task]["top5"]) / len(y_true)
                top10_acc = sum(results[task]["top10"]) / len(y_true)
                print(f"  Top-5 Accuracy: {top5_acc:.4f}")
                print(f"  Top-10 Accuracy: {top10_acc:.4f}")
            print("--------------------------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_npz", required=True, help="Path to train features NPZ")
    parser.add_argument("--val_npz", required=True, help="Path to val features NPZ")
    parser.add_argument("--vocabs", default="data/processed/protein_lm/multitask/task_vocabs.json", help="Path to task vocabs json")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out_dir", default="runs/protein_critic/checkpoints")
    args = parser.parse_args()

    train_mlp_heads(
        args.train_npz,
        args.val_npz,
        args.vocabs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        out_dir=args.out_dir
    )
