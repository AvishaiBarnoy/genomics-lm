import torch
import json
import argparse
from src.protein_lm.tokenizer import ProteinTokenizer
from src.protein_lm.models_multi import MultiTaskProteinClassifier
from src.protein_lm.config import ProteinClassifierConfig
from src.protein_lm.train_multi_task import MultiTaskProteinDataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="outputs/checkpoints/protein_critic/best_critic.pt")
    parser.add_argument("--val_data", default="data/processed/protein_lm/multitask/val.jsonl")
    args = parser.parse_args()

    tokenizer = ProteinTokenizer()
    with open("data/processed/protein_lm/multitask/task_vocabs.json", "r") as f:
        vocabs = json.load(f)
    task_dims = {
        "family": len(vocabs["pfam"]),
        "function": len(vocabs["ec"]),
        "stability": len(vocabs["stability"])
    }
    
    model_cfg = ProteinClassifierConfig(
        vocab_size=len(tokenizer.vocab),
        block_size=512,
        n_layer=4,
        n_head=4,
        n_embd=128,
        dropout=0.0,
        num_classes=0
    )
    
    model = MultiTaskProteinClassifier(model_cfg, task_dims)
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    
    val_ds = MultiTaskProteinDataset(args.val_data, tokenizer, max_length=model_cfg.block_size)
    loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    
    results = {task: {"preds": [], "targets": []} for task in task_dims}
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"]
            logits_dict = model(input_ids)
            for task in task_dims:
                targets = batch[task]
                mask = targets != -1
                if mask.any():
                    preds = torch.argmax(logits_dict[task], dim=-1)
                    results[task]["preds"].extend(preds[mask].cpu().tolist())
                    results[task]["targets"].extend(targets[mask].cpu().tolist())
                    
    for task in task_dims:
        y_true = results[task]["targets"]
        y_pred = results[task]["preds"]
        if len(y_true) > 0:
            acc = accuracy_score(y_true, y_pred)
            print(f"==================================================")
            print(f"Task: {task}")
            print(f"  Samples evaluated: {len(y_true)}")
            print(f"  Accuracy: {acc:.4f}")
            print(f"--------------------------------------------------")
            print(classification_report(y_true, y_pred, zero_division=0))
        else:
            print(f"Task: {task} has no valid validation samples.")

if __name__ == "__main__":
    main()
