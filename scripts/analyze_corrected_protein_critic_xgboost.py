"""Post-hoc class-level error analysis for the frozen XGBoost comparison.

This re-fits XGBoost with the exact selected parameters in the reference report,
then reports class support, precision/recall/F1, and confusion matrices on the
same held-out test split. It does not select new hyperparameters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from scripts.benchmark_corrected_protein_critic_xgboost import (
    INFERENCE_CONFIG_KEYS,
    critic_predictions,
    fit_xgb_with_params,
    labels,
    load_checkpoint,
    metric,
    raw_features,
    read_jsonl,
    resolve_device,
    sha256,
    verify_manifest,
)
from src.protein_lm.tokenizer import ProteinTokenizer


def class_diagnostics(y_true, prediction, names):
    class_ids = np.arange(len(names), dtype=int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, prediction, labels=class_ids, zero_division=0
    )
    confusion = confusion_matrix(y_true, prediction, labels=class_ids)
    return {
        "classes": [
            {
                "class_id": int(class_id),
                "class_name": names[class_id],
                "support": int(support[class_id]),
                "precision": float(precision[class_id]),
                "recall": float(recall[class_id]),
                "f1": float(f1[class_id]),
            }
            for class_id in class_ids
        ],
        "confusion_matrix": {
            "label_ids": class_ids.tolist(),
            "label_names": names,
            "rows_are_true_columns_are_predicted": confusion.tolist(),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/corrected_protein_critic_v1.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--critic-report", type=Path,
                        default=Path("docs/benchmarks/corrected_protein_critic_training_v1.json"))
    parser.add_argument("--benchmark-report", type=Path,
                        default=Path("docs/benchmarks/corrected_protein_critic_xgboost_v1.json"))
    parser.add_argument("--out", type=Path,
                        default=Path("docs/benchmarks/corrected_protein_critic_xgboost_class_diagnostics_v1.json"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"),
                        help="Defaults to the critic device recorded in the reference benchmark")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    try:
        import xgboost
    except ImportError as error:
        raise RuntimeError("Install the optional xgboost dependency to run this analysis") from error

    cfg = yaml.safe_load(args.config.read_text())
    base = Path(cfg["dataset_manifest"]).parent
    paths = {split: base / f"{split}.jsonl" for split in ("train", "validation", "test")}
    vocab_path = Path(cfg["task_vocabs"])
    manifest_path = Path(cfg["dataset_manifest"])
    _, verified_hashes, manifest_hash = verify_manifest(manifest_path, paths, vocab_path)

    benchmark = json.loads(args.benchmark_report.read_text())
    benchmark_sha = sha256(args.benchmark_report)
    if benchmark.get("protocol", {}).get("final_split") != "test_once":
        raise ValueError("Reference benchmark does not declare a test-once protocol")
    if benchmark.get("manifest_sha256") != manifest_hash:
        raise ValueError("Reference benchmark uses a different dataset manifest")
    split_hashes = {role: verified_hashes[role] for role in paths}
    if benchmark.get("split_sha256") != split_hashes:
        raise ValueError("Reference benchmark uses different split artifacts")

    training_report = json.loads(args.critic_report.read_text())
    expected_checkpoint = training_report.get("checkpoint_sha256", {}).get("best")
    if not expected_checkpoint or sha256(args.checkpoint) != expected_checkpoint:
        raise ValueError("Checkpoint does not match the report-selected best critic")
    if benchmark.get("checkpoint", {}).get("sha256") != expected_checkpoint:
        raise ValueError("Reference benchmark was produced with a different critic checkpoint")

    tokenizer = ProteinTokenizer()
    vocabs = json.loads(vocab_path.read_text())
    device_name = args.device or benchmark.get("critic_device")
    if not device_name:
        raise ValueError("Reference benchmark does not record the critic inference device")
    device = resolve_device(device_name)
    if str(device) != benchmark["critic_device"]:
        raise ValueError("Use the same critic inference device as the reference benchmark")
    model, checkpoint = load_checkpoint(args.checkpoint, cfg, vocabs, tokenizer, device)
    provenance = checkpoint["dataset_provenance"]
    if provenance.get("manifest", {}).get("sha256") != manifest_hash:
        raise ValueError("Checkpoint provenance uses a different dataset manifest")
    for role in (*paths.keys(), "task_vocabs"):
        key = "validation" if role == "validation" else role
        if provenance["artifacts"][key]["sha256"] != verified_hashes[role]:
            raise ValueError(f"Checkpoint provenance differs for {role}")
    for key in INFERENCE_CONFIG_KEYS:
        if benchmark.get("critic_inference_config", {}).get(key) != checkpoint["cfg"][key]:
            raise ValueError(f"Reference benchmark inference config differs for {key}")
    records = {role: read_jsonl(path) for role, path in paths.items()}
    max_residues = model.config.block_size - 2

    raw_train = {task: raw_features(records["train"], task == "stability", max_residues)
                 for task in ("pfam", "ec", "stability")}
    raw_test = {task: raw_features(records["test"], task == "stability", max_residues)
                for task in ("pfam", "ec", "stability")}
    critic_test = critic_predictions(model, paths["test"], tokenizer, cfg, args.batch_size, device)
    y_train = {task: labels(records["train"], task) for task in ("pfam", "ec", "stability")}
    y_test = {task: labels(records["test"], task) for task in ("pfam", "ec", "stability")}

    output = {
        "schema_version": 1,
        "analysis": "posthoc_class_level_decomposition_of_frozen_xgboost_v1_comparison",
        "scope": "per_class_raw_sequence_xgboost_vs_protein_critic; embedding_xgboost_remains_aggregate_only",
        "not_an_independent_replication": True,
        "reference_benchmark_report": str(args.benchmark_report),
        "reference_benchmark_report_sha256": benchmark_sha,
        "xgboost_version": xgboost.__version__,
        "xgboost_seed": args.seed,
        "checkpoint_sha256": expected_checkpoint,
        "manifest_sha256": manifest_hash,
        "split_sha256": split_hashes,
        "task_vocabs_sha256": verified_hashes["task_vocabs"],
        "test_predictions_recomputed_without_test_based_selection": True,
        "critic_device": str(device),
        "tasks": {},
    }

    for task in ("pfam", "ec", "stability"):
        task_result = {"n_train_labelled": int(
            np.isfinite(y_train[task]).sum() if task == "stability" else (y_train[task] >= 0).sum()
        )}
        test_mask = np.isfinite(y_test[task]) if task == "stability" else y_test[task] >= 0
        test_y = y_test[task][test_mask]
        test_groups = np.asarray([
            row.get("protein_cluster", row["record_id"])
            for row, keep in zip(records["test"], test_mask) if keep
        ])
        task_result["n_test_labelled"] = int(test_mask.sum())
        task_result["n_test_clusters"] = int(np.unique(test_groups).size)
        task_result["test_metric"] = "mae" if task == "stability" else "balanced_accuracy"
        task_result["models"] = {}

        name = "raw_sequence"
        reference = benchmark["tasks"][task][name]
        selected_params = reference["selected_params"]
        estimator, fit_count = fit_xgb_with_params(
            task, raw_train[task], y_train[task], selected_params, args.seed
        )
        prediction = estimator.predict(raw_test[task][test_mask])
        if task == "stability":
            model_result = {
                "selected_params_reused_from_reference": selected_params,
                "n_train_labelled": fit_count,
                "test_mae": metric(task, test_y, prediction),
                "class_diagnostics": None,
            }
        else:
            id_to_name = {int(index): label for label, index in vocabs[task].items()}
            names = [id_to_name[class_id] for class_id in range(len(id_to_name))]
            macro_f1 = float(f1_score(test_y, prediction, average="macro", zero_division=0))
            if not np.isclose(macro_f1, reference["test_macro_f1"], rtol=0, atol=1e-12):
                raise ValueError(f"Recomputed {task}/{name} macro-F1 differs from the reference report")
            model_result = {
                "selected_params_reused_from_reference": selected_params,
                "n_train_labelled": fit_count,
                "test_balanced_accuracy": metric(task, test_y, prediction),
                "test_macro_f1": macro_f1,
                "class_diagnostics": class_diagnostics(test_y, prediction, names),
            }
        score_key = "test_mae" if task == "stability" else "test_balanced_accuracy"
        if not np.isclose(model_result[score_key], reference["test_score"], rtol=0, atol=1e-12):
            raise ValueError(f"Recomputed {task}/{name} metric differs from the reference report")
        model_result["matches_reference_aggregate_score"] = True
        task_result["models"][name] = model_result

        critic_prediction = critic_test[task][test_mask]
        reference_critic = benchmark["tasks"][task]["protein_critic"]["metrics"]["test"]
        critic_score_key = "mae" if task == "stability" else "balanced_accuracy"
        critic_score = metric(task, test_y, critic_prediction)
        if not np.isclose(critic_score, reference_critic[critic_score_key], rtol=0, atol=1e-12):
            raise ValueError(f"Recomputed {task} critic metric differs from the reference report")
        if task == "stability":
            task_result["models"]["protein_critic"] = {
                "test_mae": critic_score,
                "matches_reference_aggregate_score": True,
                "class_diagnostics": None,
            }
        else:
            id_to_name = {int(index): label for label, index in vocabs[task].items()}
            names = [id_to_name[class_id] for class_id in range(len(id_to_name))]
            critic_macro_f1 = float(f1_score(test_y, critic_prediction, average="macro", zero_division=0))
            if not np.isclose(critic_macro_f1, reference_critic["macro_f1"], rtol=0, atol=1e-12):
                raise ValueError(f"Recomputed {task} critic macro-F1 differs from the reference report")
            task_result["models"]["protein_critic"] = {
                "test_balanced_accuracy": critic_score,
                "test_macro_f1": critic_macro_f1,
                "matches_reference_aggregate_score": True,
                "class_diagnostics": class_diagnostics(test_y, critic_prediction, names),
            }
        output["tasks"][task] = task_result

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"output": str(args.out), "reference_benchmark_sha256": benchmark_sha,
                      "selection": "reused from reference report; no test-based tuning"}, indent=2))


if __name__ == "__main__":
    main()
