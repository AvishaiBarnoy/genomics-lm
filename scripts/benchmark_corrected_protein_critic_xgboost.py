"""Compare the frozen corrected ProteinCritic with matched XGBoost controls.

Hyperparameters are selected on validation only. Test metrics are computed once,
after model selection, and are never used to choose a checkpoint or parameters.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import balanced_accuracy_score, f1_score, mean_absolute_error
from torch.utils.data import DataLoader

from src.protein_lm.config import ProteinClassifierConfig
from src.protein_lm.dataset import MultiTaskProteinDataset, collate_protein_batch
from src.protein_lm.models_multi import MultiTaskProteinClassifier
from src.protein_lm.tokenizer import ProteinTokenizer


AA = "ARNDCEQGHILKMFPSTWYV"
AA_INDEX = {aa: i for i, aa in enumerate(AA)}
HYDROPATHY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def verify_manifest(manifest_path: Path, split_paths: dict[str, Path], vocab_path: Path):
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("protocol") != "mmseqs_cluster_held_out_multitask_protein_critic":
        raise ValueError("Unexpected corrected-critic dataset protocol")
    for role, path in {**split_paths, "task_vocabs": vocab_path}.items():
        key = "validation" if role == "validation" else role
        expected = manifest["artifacts"][key]["sha256"]
        if sha256(path) != expected:
            raise ValueError(f"{role} artifact SHA-256 differs from dataset manifest")
    return manifest


def raw_features(records: list[dict], stability: bool, max_residues: int) -> np.ndarray:
    """Fixed, label-free features; all residue n-grams are normalized frequencies."""
    width = 1 + 20 + 20**2 + 20**3 + (5 if stability else 0)
    matrix = np.zeros((len(records), width), dtype=np.float32)
    for row, record in enumerate(records):
        # Match the critic's BOS/EOS-aware truncation window exactly.
        sequence = record["sequence"].upper()[:max_residues]
        matrix[row, 0] = math.log1p(len(sequence))
        counts = np.zeros(20, dtype=np.float32)
        for residue in sequence:
            if residue in AA_INDEX:
                counts[AA_INDEX[residue]] += 1
        if len(sequence):
            counts /= len(sequence)
        matrix[row, 1:21] = counts
        offset = 21
        for n in (2, 3):
            size = 20**n
            grams = matrix[row, offset:offset + size]
            total = 0
            for start in range(max(0, len(sequence) - n + 1)):
                gram = sequence[start:start + n]
                if all(letter in AA_INDEX for letter in gram):
                    index = 0
                    for letter in gram:
                        index = index * 20 + AA_INDEX[letter]
                    grams[index] += 1
                    total += 1
            if total:
                grams /= total
            offset += size
        if stability:
            known = [residue for residue in sequence if residue in AA_INDEX]
            denominator = max(1, len(known))
            charge = sum(1 if r in "KR" else -1 if r in "DE" else 0 for r in known)
            matrix[row, offset:] = (
                sum(HYDROPATHY[r] for r in known) / denominator,
                charge / denominator,
                sum(r in "FWY" for r in known) / denominator,
                sum(r in "ILV" for r in known) / denominator,
                sum(r == "C" for r in known) / denominator,
            )
    return matrix


def labels(records: list[dict], task: str) -> np.ndarray:
    key = {"pfam": "pfam_id", "ec": "ec_id", "stability": "stability_score"}[task]
    values = [record.get(key) for record in records]
    if task == "stability":
        return np.asarray([np.nan if value is None else float(value) for value in values])
    return np.asarray([-1 if value is None else int(value) for value in values], dtype=int)


def load_checkpoint(path: Path, cfg: dict, vocabs: dict, tokenizer: ProteinTokenizer, device):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or state.get("dataset_provenance", {}).get("status") != "manifest_verified":
        raise ValueError("Checkpoint lacks manifest-verified dataset provenance")
    dims = {"family": len(vocabs["pfam"]), "function": len(vocabs["ec"]), "stability": 1}
    model_cfg = ProteinClassifierConfig(
        vocab_size=len(tokenizer.vocab), block_size=cfg["block_size"],
        n_layer=cfg["n_layer"], n_head=cfg["n_head"], n_embd=cfg["n_embd"],
        dropout=0.0, num_classes=0, pooling=cfg["pooling"],
        bidirectional=cfg["bidirectional"],
    )
    model = MultiTaskProteinClassifier(model_cfg, dims)
    weights = state.get("model_state_dict", state)
    weights = weights.get("model", weights)
    model.load_state_dict(weights, strict=True)
    model.to(device).eval()
    return model, state


@torch.inference_mode()
def embeddings(model, path: Path, tokenizer, cfg, batch_size, device):
    dataset = MultiTaskProteinDataset(
        str(path), tokenizer, max_length=model.config.block_size,
        dynamic_padding=True,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=lambda batch: collate_protein_batch(batch, tokenizer.pad_token_id),
    )
    vectors = []
    for batch in loader:
        vectors.append(model.extract_latent(
            batch["input_ids"].to(device), batch["attention_mask"].to(device)
        ).cpu().numpy())
    return np.concatenate(vectors, axis=0)


@torch.inference_mode()
def critic_predictions(model, path: Path, tokenizer, cfg, batch_size, device):
    dataset = MultiTaskProteinDataset(
        str(path), tokenizer, max_length=model.config.block_size,
        dynamic_padding=True,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=lambda batch: collate_protein_batch(batch, tokenizer.pad_token_id),
    )
    predictions = {"pfam": [], "ec": [], "stability": []}
    for batch in loader:
        inputs = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        output = model(inputs, attention_mask=mask)
        predictions["pfam"].extend(output["family"].argmax(dim=-1).cpu().tolist())
        predictions["ec"].extend(output["function"].argmax(dim=-1).cpu().tolist())
        predictions["stability"].extend(output["stability"].squeeze(-1).cpu().tolist())
    return {task: np.asarray(value) for task, value in predictions.items()}


def metric(task: str, y: np.ndarray, pred: np.ndarray) -> float:
    if task == "stability":
        return float(mean_absolute_error(y, pred))
    return float(balanced_accuracy_score(y, pred))


def tune_xgb(task, x_train, y_train, x_val, y_val, seed):
    try:
        from xgboost import XGBClassifier, XGBRegressor
    except ImportError as error:
        raise RuntimeError("Install the optional xgboost dependency to run this benchmark") from error
    valid_train = np.isfinite(y_train) if task == "stability" else y_train >= 0
    valid_val = np.isfinite(y_val) if task == "stability" else y_val >= 0
    train_x, train_y = x_train[valid_train], y_train[valid_train]
    val_x, val_y = x_val[valid_val], y_val[valid_val]
    if len(train_y) == 0 or len(val_y) == 0:
        raise ValueError(f"No labelled train/validation records for {task}")
    best = None
    grid = itertools.product((3, 6), (200, 500))
    for depth, trees in grid:
        args = dict(n_estimators=trees, max_depth=depth, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                    n_jobs=1, random_state=seed, tree_method="hist")
        if task == "stability":
            model = XGBRegressor(objective="reg:squarederror", **args)
        else:
            model = XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", **args)
        model.fit(train_x, train_y)
        prediction = model.predict(val_x)
        score = metric(task, val_y, prediction)
        rank_score = -score if task == "stability" else score
        if best is None or rank_score > best[0]:
            best = (rank_score, {"max_depth": depth, "n_estimators": trees}, model, score)
    return best[1], best[2], best[3], int(valid_train.sum()), int(valid_val.sum())


def bootstrap_ci(task, y, pred, groups, seed, replicates=1000):
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        selected = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == group) for group in selected])
        sample_y, sample_pred = y[indices], pred[indices]
        if task != "stability" and len(np.unique(sample_y)) < 2:
            continue
        values.append(metric(task, sample_y, sample_pred))
    if not values:
        return [None, None]
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def paired_improvement_ci(task, y, candidate, critic, groups, seed, replicates=1000):
    """Positive means candidate is better (higher balanced accuracy/lower MAE)."""
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        selected = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == group) for group in selected])
        sample_y = y[indices]
        if task != "stability" and len(np.unique(sample_y)) < 2:
            continue
        delta = metric(task, sample_y, candidate[indices]) - metric(task, sample_y, critic[indices])
        values.append(-delta if task == "stability" else delta)
    if not values:
        return [None, None]
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/corrected_protein_critic_v1.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True, help="Exact frozen corrected critic checkpoint")
    parser.add_argument("--critic-report", type=Path, default=Path("docs/benchmarks/corrected_protein_critic_training_v1.json"),
                        help="Training report that records the selected best checkpoint SHA-256")
    parser.add_argument("--out", type=Path, default=Path("docs/benchmarks/corrected_protein_critic_xgboost_v1.json"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    args = parser.parse_args()
    try:
        import xgboost
    except ImportError as error:
        raise RuntimeError("Install the optional xgboost dependency to run this benchmark") from error
    cfg = yaml.safe_load(args.config.read_text())
    base = Path(cfg["dataset_manifest"]).parent
    paths = {split: base / f"{name}.jsonl" for split, name in
             (("train", "train"), ("validation", "validation"), ("test", "test"))}
    vocab_path = Path(cfg["task_vocabs"])
    manifest = verify_manifest(Path(cfg["dataset_manifest"]), paths, vocab_path)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    report = json.loads(args.critic_report.read_text())
    expected_hash = report.get("checkpoint_sha256", {}).get("best")
    if not expected_hash:
        raise ValueError("Critic report does not identify the selected best checkpoint")
    checkpoint_hash = sha256(args.checkpoint)
    if checkpoint_hash != expected_hash:
        raise ValueError("Selected checkpoint is not the report's frozen best checkpoint")
    tokenizer = ProteinTokenizer()
    vocabs = json.loads(vocab_path.read_text())
    device = torch.device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, cfg, vocabs, tokenizer, device)
    expected_checkpoint = checkpoint.get("checkpoint_sha256")
    if expected_checkpoint and expected_checkpoint != checkpoint_hash:
        raise ValueError("Checkpoint's embedded SHA-256 does not match the selected file")
    for role, path in paths.items():
        key = "validation" if role == "validation" else role
        if checkpoint["dataset_provenance"]["artifacts"][key]["sha256"] != manifest["artifacts"][key]["sha256"]:
            raise ValueError(f"Checkpoint provenance differs for {role} split")

    records = {role: read_jsonl(path) for role, path in paths.items()}
    max_residues = cfg["block_size"] - 2
    raw = {role: raw_features(rows, stability=False, max_residues=max_residues)
           for role, rows in records.items()}
    raw_stability = {role: raw_features(rows, stability=True, max_residues=max_residues)
                     for role, rows in records.items()}
    frozen = {role: embeddings(model, paths[role], tokenizer, cfg, args.batch_size, device)
              for role in ("train", "validation", "test")}
    critic_outputs = {
        role: critic_predictions(model, paths[role], tokenizer, cfg, args.batch_size, device)
        for role in ("validation", "test")
    }
    results = {"schema_version": 1, "dataset_id": cfg["critic_training_contract"]["dataset_id"],
               "checkpoint": {"path": str(args.checkpoint), "sha256": checkpoint_hash},
               "split_sha256": {role: sha256(path) for role, path in paths.items()},
               "protocol": {"selection_split": "validation", "final_split": "test_once",
                            "bootstrap_unit": "protein_cluster", "raw_features":
                            "log_length, amino-acid composition, normalized dipeptide and tripeptide frequencies",
                            "stability_additional_features":
                            "mean Kyte-Doolittle hydropathy, net charge/residue, aromatic fraction, aliphatic fraction, cysteine fraction"},
               "tasks": {}}
    for task in ("pfam", "ec", "stability"):
        task_result = {}
        features = {"raw_sequence": raw_stability if task == "stability" else raw,
                    "frozen_critic_embedding": frozen}
        y = {role: labels(rows, task) for role, rows in records.items()}
        for name, matrices in features.items():
            best_params, _, validation_score, train_count, validation_count = tune_xgb(
                task, matrices["train"], y["train"], matrices["validation"],
                y["validation"], args.seed)
            fit_mask = np.isfinite(y["train"]) if task == "stability" else y["train"] >= 0
            fit_val = np.isfinite(y["validation"]) if task == "stability" else y["validation"] >= 0
            fit_x = np.concatenate((matrices["train"][fit_mask], matrices["validation"][fit_val]))
            fit_y = np.concatenate((y["train"][fit_mask], y["validation"][fit_val]))
            try:
                from xgboost import XGBClassifier, XGBRegressor
            except ImportError as error:
                raise RuntimeError("Install the optional xgboost dependency to run this benchmark") from error
            estimator_args = dict(**best_params, learning_rate=0.05, subsample=0.8,
                                  colsample_bytree=0.8, reg_lambda=1.0, n_jobs=1,
                                  random_state=args.seed, tree_method="hist")
            if task == "stability":
                estimator = XGBRegressor(objective="reg:squarederror", **estimator_args)
            else:
                estimator = XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", **estimator_args)
            estimator.fit(fit_x, fit_y)
            test_y = y["test"]
            mask = np.isfinite(test_y) if task == "stability" else test_y >= 0
            prediction = estimator.predict(matrices["test"][mask])
            test_groups = np.asarray([row.get("protein_cluster", row["record_id"])
                                      for row, keep in zip(records["test"], mask) if keep])
            critic_test = critic_outputs["test"][task][mask]
            xgb_score = metric(task, test_y[mask], prediction)
            critic_score = metric(task, test_y[mask], critic_test)
            improvement = critic_score - xgb_score if task == "stability" else xgb_score - critic_score
            task_result[name] = {
                "selected_params": best_params,
                "n_train_labelled": train_count,
                "n_validation_labelled": validation_count,
                "n_test_labelled": int(mask.sum()),
                "validation_selection_metric": "mae_minimized" if task == "stability" else "balanced_accuracy_maximized",
                "selected_validation_score": validation_score,
                "test_metric": "mae" if task == "stability" else "balanced_accuracy",
                "test_score": xgb_score,
                "test_score_95pct_cluster_bootstrap_ci": bootstrap_ci(
                    task, test_y[mask], prediction, test_groups, args.seed,
                    args.bootstrap_replicates),
                "test_improvement_over_critic": improvement,
                "test_improvement_over_critic_95pct_paired_cluster_bootstrap_ci":
                    paired_improvement_ci(task, test_y[mask], prediction, critic_test,
                                          test_groups, args.seed, args.bootstrap_replicates),
                "test_macro_f1": None if task == "stability" else float(
                    f1_score(test_y[mask], prediction, average="macro", zero_division=0)),
            }
        critic_metrics = {}
        for role in ("validation", "test"):
            target = y[role]
            mask = np.isfinite(target) if task == "stability" else target >= 0
            pred = critic_outputs[role][task][mask]
            metric_name = "mae" if task == "stability" else "balanced_accuracy"
            critic_metrics[role] = {
                "n_labelled": int(mask.sum()),
                metric_name: metric(task, target[mask], pred),
                "macro_f1": None if task == "stability" else float(
                    f1_score(target[mask], pred, average="macro", zero_division=0)),
            }
            if role == "test":
                role_records = records[role]
                groups = np.asarray([row.get("protein_cluster", row["record_id"])
                                     for row, keep in zip(role_records, mask) if keep])
                critic_metrics[role][f"{metric_name}_95pct_cluster_bootstrap_ci"] = bootstrap_ci(
                    task, target[mask], pred, groups, args.seed, args.bootstrap_replicates)
        task_result["protein_critic"] = {
            "checkpoint_sha256": checkpoint_hash,
            "metrics": critic_metrics,
        }
        results["tasks"][task] = task_result
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results["xgboost_version"] = xgboost.__version__
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({"output": str(args.out), "checkpoint_sha256": checkpoint_hash,
                      "test_evaluations": "completed once per modality and task"}, indent=2))


if __name__ == "__main__":
    main()
