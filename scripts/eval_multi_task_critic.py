import torch
import json
import argparse
import yaml
import os
import numpy as np
from src.protein_lm.tokenizer import ProteinTokenizer
from src.protein_lm.models_multi import MultiTaskProteinClassifier
from src.protein_lm.config import ProteinClassifierConfig
from src.protein_lm.train_multi_task import (
    LengthBucketBatchSampler,
    MultiTaskProteinDataset,
    collate_protein_batch,
)
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
)


def _device(name):
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _ordered_vocab(vocab):
    return [name for name, _ in sorted(vocab.items(), key=lambda item: item[1])]


def _safe_float(value):
    if value is None or np.isnan(value):
        return None
    return float(value)


def parse_float_list(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def threshold_metrics(y_true, y_prob, thresholds):
    rows = []
    for threshold in thresholds:
        pred = y_prob >= threshold
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, pred, average="binary", zero_division=0
        )
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "predicted_fraction": float(pred.mean()),
            }
        )
    return rows


def top_fraction_enrichment(y_true, y_prob, fractions):
    rows = []
    prevalence = float(y_true.mean())
    order = np.argsort(-y_prob)
    for fraction in fractions:
        k = max(1, int(np.ceil(len(y_true) * fraction)))
        selected = y_true[order[:k]]
        selected_positive_rate = float(selected.mean())
        enrichment = (
            selected_positive_rate / prevalence if prevalence > 0.0 else np.nan
        )
        rows.append(
            {
                "fraction": float(fraction),
                "k": int(k),
                "positive_rate": selected_positive_rate,
                "enrichment": _safe_float(enrichment),
                "positives": int(selected.sum()),
            }
        )
    return rows


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
    parser.add_argument("--task_vocabs", default=None, help="Optional task vocab JSON")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda")
    parser.add_argument("--out_json", default=None, help="Optional path for metrics JSON")
    parser.add_argument(
        "--thresholds",
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
        help="Comma-separated thresholds for multi-label calibration metrics",
    )
    parser.add_argument(
        "--top_fractions",
        default="0.01,0.05,0.1",
        help="Comma-separated validation fractions for top-k enrichment metrics",
    )
    args = parser.parse_args()
    thresholds = parse_float_list(args.thresholds)
    top_fractions = parse_float_list(args.top_fractions)

    # Load YAML configuration if it exists
    cfg = {}
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
    if args.val_data == parser.get_default("val_data") and cfg.get("val_data"):
        args.val_data = cfg["val_data"]
    if args.task_vocabs is None:
        args.task_vocabs = cfg.get("task_vocabs")

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
    multi_label_tasks = list(cfg.get("multi_label_tasks", []))
    for task in multi_label_tasks:
        key = f"heads.{task}.weight"
        if key in state_dict:
            task_dims[task] = state_dict[key].shape[0]

    vocab_labels = {}
    if args.task_vocabs:
        with open(args.task_vocabs, "r") as f:
            vocabs = json.load(f)
        for task in multi_label_tasks:
            if task in vocabs:
                vocab_labels[task] = _ordered_vocab(vocabs[task])

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
    device = _device(args.device)
    model.to(device)
    model.eval()

    val_ds = MultiTaskProteinDataset(
        args.val_data,
        tokenizer,
        max_length=model_cfg.block_size,
        dynamic_padding=bool(cfg.get("dynamic_padding", False)),
        multi_label_tasks=multi_label_tasks,
    )
    if cfg.get("dynamic_padding", False):
        loader = DataLoader(
            val_ds,
            batch_sampler=LengthBucketBatchSampler(
                val_ds, args.batch_size, shuffle=False
            ),
            collate_fn=lambda batch: collate_protein_batch(
                batch, tokenizer.pad_token_id
            ),
        )
    else:
        loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    results = {
        task: {"preds": [], "targets": [], "top5": [], "top10": []}
        for task in task_dims
        if task not in multi_label_tasks
    }
    multi_label_results = {
        task: {"targets": [], "probs": []} for task in multi_label_tasks
    }
    bce = torch.nn.BCEWithLogitsLoss(reduction="sum")
    bce_sum = {task: 0.0 for task in multi_label_tasks}
    bce_items = {task: 0 for task in multi_label_tasks}

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            logits_dict = model(input_ids, attention_mask=attention_mask)
            for task in results:
                targets = batch[task]
                mask = targets != -1
                if mask.any():
                    logits = logits_dict[task].cpu()[mask]
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
            for task in multi_label_tasks:
                if task not in logits_dict:
                    continue
                targets = batch[task].to(device)
                logits = logits_dict[task]
                bce_sum[task] += bce(logits, targets).item()
                bce_items[task] += targets.numel()
                multi_label_results[task]["targets"].append(targets.cpu().numpy())
                multi_label_results[task]["probs"].append(
                    torch.sigmoid(logits).cpu().numpy()
                )

    summary = {"single_label": {}, "multi_label": {}}
    for task in task_dims:
        if task in multi_label_tasks:
            continue
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
            summary["single_label"][task] = {
                "samples": len(y_true),
                "top1_accuracy": float(acc),
            }
            print("--------------------------------------------------")
            print(classification_report(y_true, y_pred, zero_division=0))
        else:
            print(f"Task: {task} has no valid validation samples.")

    for task, data in multi_label_results.items():
        if not data["targets"]:
            print(f"Task: {task} has no valid validation samples.")
            continue
        y_true = np.concatenate(data["targets"], axis=0)
        y_prob = np.concatenate(data["probs"], axis=0)
        labels = vocab_labels.get(task, [str(i) for i in range(y_true.shape[1])])
        task_bce = bce_sum[task] / max(bce_items[task], 1)
        prevalence = y_true.mean(axis=0)
        clipped = np.clip(prevalence, 1e-6, 1 - 1e-6)
        baseline_bce = -np.mean(
            y_true * np.log(clipped) + (1 - y_true) * np.log(1 - clipped)
        )
        print("==================================================")
        print(f"Task: {task}")
        print(f"  Samples evaluated: {y_true.shape[0]}")
        print(f"  BCE Loss: {task_bce:.4f}")
        print(f"  Prevalence BCE baseline: {baseline_bce:.4f}")
        summary["multi_label"][task] = {
            "samples": int(y_true.shape[0]),
            "bce": float(task_bce),
            "prevalence_bce_baseline": float(baseline_bce),
            "labels": {},
        }
        print("--------------------------------------------------")
        print("label,prevalence,mean_prob,ap,roc_auc,precision@0.5,recall@0.5,f1@0.5")
        for i, label in enumerate(labels):
            yt = y_true[:, i]
            yp = y_prob[:, i]
            if yt.min() == yt.max():
                ap = np.nan
                auc = np.nan
            else:
                ap = average_precision_score(yt, yp)
                auc = roc_auc_score(yt, yp)
            pred = yp >= 0.5
            precision, recall, f1, _ = precision_recall_fscore_support(
                yt, pred, average="binary", zero_division=0
            )
            print(
                f"{label},{yt.mean():.4f},{yp.mean():.4f},"
                f"{ap:.4f},{auc:.4f},{precision:.4f},{recall:.4f},{f1:.4f}"
            )
            threshold_rows = threshold_metrics(yt, yp, thresholds)
            enrichment_rows = top_fraction_enrichment(yt, yp, top_fractions)
            print("  thresholds:")
            for row in threshold_rows:
                print(
                    "    "
                    f"t={row['threshold']:.2f},"
                    f"precision={row['precision']:.4f},"
                    f"recall={row['recall']:.4f},"
                    f"f1={row['f1']:.4f},"
                    f"predicted_fraction={row['predicted_fraction']:.4f}"
                )
            print("  top_fraction_enrichment:")
            for row in enrichment_rows:
                enrichment = row["enrichment"]
                enrichment_text = "nan" if enrichment is None else f"{enrichment:.2f}"
                print(
                    "    "
                    f"top={row['fraction']:.2%},"
                    f"k={row['k']},"
                    f"positive_rate={row['positive_rate']:.4f},"
                    f"enrichment={enrichment_text}x,"
                    f"positives={row['positives']}"
                )
            summary["multi_label"][task]["labels"][label] = {
                "prevalence": float(yt.mean()),
                "mean_probability": float(yp.mean()),
                "average_precision": _safe_float(ap),
                "roc_auc": _safe_float(auc),
                "precision_at_0_5": float(precision),
                "recall_at_0_5": float(recall),
                "f1_at_0_5": float(f1),
                "thresholds": threshold_rows,
                "top_fraction_enrichment": enrichment_rows,
            }

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
