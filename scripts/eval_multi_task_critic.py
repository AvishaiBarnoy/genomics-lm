import torch
import json
import argparse
import yaml
import os
from src.protein_lm.tokenizer import ProteinTokenizer
from src.protein_lm.models_multi import MultiTaskProteinClassifier
from src.protein_lm.config import ProteinClassifierConfig
from src.protein_lm.train_multi_task import MultiTaskProteinDataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/protein_critic.yaml", help="Path to YAML config"
    )
    parser.add_argument(
        "--ckpt", default="outputs/checkpoints/protein_critic/best_critic.pt"
    )
    parser.add_argument(
        "--val_data", default="data/processed/protein_lm/multitask/val.jsonl"
    )
    args = parser.parse_args()

    # Load YAML configuration if it exists
    cfg = {}
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)

    tokenizer = ProteinTokenizer()
    state = torch.load(args.ckpt, map_location="cpu")
    state_dict = state.get("model_state_dict", state)
    state_dict = state_dict.get("model", state_dict)

    # Dynamically infer task dimensions from checkpoint weights
    task_dims = {}
    if "heads.family.weight" in state_dict:
        task_dims["family"] = state_dict["heads.family.weight"].shape[0]
    if "heads.function.weight" in state_dict:
        task_dims["function"] = state_dict["heads.function.weight"].shape[0]
    if "heads.stability.weight" in state_dict:
        task_dims["stability"] = state_dict["heads.stability.weight"].shape[0]

    model_cfg = ProteinClassifierConfig(
        vocab_size=len(tokenizer.vocab),
        block_size=cfg.get("block_size", 512),
        n_layer=cfg.get("n_layer", 4),
        n_head=cfg.get("n_head", 4),
        n_embd=cfg.get("n_embd", 128),
        dropout=0.0,
        num_classes=0,
    )

    model = MultiTaskProteinClassifier(model_cfg, task_dims)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    val_ds = MultiTaskProteinDataset(
        args.val_data, tokenizer, max_length=model_cfg.block_size
    )
    loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    results = {
        task: {"preds": [], "targets": [], "top5": [], "top10": []}
        for task in task_dims
    }

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"]
            logits_dict = model(input_ids)
            for task in task_dims:
                targets = batch[task]
                mask = targets != -1
                if mask.any():
                    logits = logits_dict[task][mask]
                    gold = targets[mask]

                    # Top-1
                    preds = torch.argmax(logits, dim=-1)
                    results[task]["preds"].extend(preds.cpu().tolist())
                    results[task]["targets"].extend(gold.cpu().tolist())

                    # Top-5 and Top-10
                    num_classes = logits.size(-1)
                    k_val = min(10, num_classes)
                    _, topk_idx = torch.topk(logits, k=k_val, dim=-1)
                    correct_topk = topk_idx == gold.unsqueeze(-1)

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
            print(classification_report(y_true, y_pred, zero_division=0))
        else:
            print(f"Task: {task} has no valid validation samples.")


if __name__ == "__main__":
    main()
