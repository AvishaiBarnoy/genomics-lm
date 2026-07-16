from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as skm
import yaml


def load_yaml(path: str | Path) -> dict:
    """Loads a YAML configuration file from path."""
    return yaml.safe_load(Path(path).read_text()) or {}


def ensure_dir(p: str | Path) -> Path:
    """Ensures directory exists and returns its Path object."""
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: str | Path, data: dict) -> None:
    """Saves a dictionary to JSON format at path."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True))


def _one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Helper to convert integer class labels into a one-hot representation."""
    out = np.zeros((y.shape[0], n_classes), dtype=np.float32)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    bootstrap: bool = False,
    n_resamples: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    """Computes standard and robust evaluation metrics (accuracy, macro_f1, balanced_accuracy, auroc, macro_auprc) and optional bootstrapped 95% CIs."""
    metrics: Dict[str, float] = {}

    def _calc(true, pred, proba):
        res = {}
        res["accuracy"] = float(skm.accuracy_score(true, pred))
        res["balanced_accuracy"] = float(skm.balanced_accuracy_score(true, pred))
        try:
            res["macro_f1"] = float(skm.f1_score(true, pred, average="macro"))
        except Exception:
            pass
            
        if proba is not None:
            try:
                if proba.ndim == 1:
                    res["auroc"] = float(skm.roc_auc_score(true, proba))
                    res["macro_auprc"] = float(skm.average_precision_score(true, proba))
                else:
                    n_classes = proba.shape[1]
                    true_oh = _one_hot(true, n_classes)
                    res["auroc"] = float(skm.roc_auc_score(true_oh, proba, multi_class="ovr"))
                    res["macro_auprc"] = float(skm.average_precision_score(true_oh, proba, average="macro"))
            except Exception:
                pass
        return res

    # Compute point estimates
    point_estimates = _calc(y_true, y_pred, y_proba)
    metrics.update(point_estimates)

    # Compute bootstrap confidence intervals
    if bootstrap and len(y_true) > 0:
        rng = np.random.default_rng(seed)
        bootstrap_runs = []
        for _ in range(n_resamples):
            indices = rng.choice(len(y_true), size=len(y_true), replace=True)
            y_true_b = y_true[indices]
            y_pred_b = y_pred[indices]
            y_proba_b = y_proba[indices] if y_proba is not None else None
            try:
                bootstrap_runs.append(_calc(y_true_b, y_pred_b, y_proba_b))
            except Exception:
                pass

        if bootstrap_runs:
            for key in point_estimates:
                vals = sorted([run[key] for run in bootstrap_runs if key in run])
                if vals:
                    low_idx = int(len(vals) * 0.025)
                    high_idx = int(len(vals) * 0.975)
                    metrics[f"{key}_ci_lower"] = float(vals[low_idx])
                    metrics[f"{key}_ci_upper"] = float(vals[high_idx])
                    
    return metrics


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, out_path: str | Path, normalize: bool = True) -> None:
    """Plots and saves the confusion matrix for model predictions."""
    labels = np.unique(np.concatenate([y_true, y_pred]))
    cm = skm.confusion_matrix(y_true, y_pred, labels=labels, normalize=("true" if normalize else None))
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Confusion Matrix" + (" (normalized)" if normalize else ""))
    plt.tight_layout()
    ensure_dir(Path(out_path).parent)
    plt.savefig(out_path)
    plt.close(fig)


def plot_calibration(y_true: np.ndarray, y_proba: np.ndarray, out_path: str | Path, n_bins: int = 10) -> None:
    """Plots and saves the calibration reliability curve for model probability estimates."""
    # Supports binary or per-class reliability diagram (plots first class otherwise)
    if y_proba.ndim > 1 and y_proba.shape[1] > 1:
        scores = y_proba.max(axis=1)
        y_bin = (y_true == y_proba.argmax(axis=1)).astype(int)
    else:
        scores = y_proba.reshape(-1)
        y_bin = y_true
    if np.any(scores < 0.0) or np.any(scores > 1.0):
        scores = 1.0 / (1.0 + np.exp(-scores))
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(y_bin, scores, n_bins=n_bins)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(prob_pred, prob_true, marker="o", label="model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("True frequency")
    ax.set_title("Calibration curve")
    ax.legend()
    plt.tight_layout()
    ensure_dir(Path(out_path).parent)
    plt.savefig(out_path)
    plt.close(fig)


def save_npz(path: str | Path, **arrays) -> None:
    """Saves arrays to a compressed npz file at path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **arrays)


def load_npz(path: str | Path) -> Dict[str, np.ndarray]:
    """Loads and returns arrays from an npz file."""
    with np.load(path, allow_pickle=True) as blob:
        return {k: blob[k] for k in blob.files}


@dataclass
class EmbeddingPack:
    """A dataclass holding sequence embeddings and optional labels."""
    X: np.ndarray
    y: Optional[np.ndarray] = None

    ids: Optional[List[str]] = None
