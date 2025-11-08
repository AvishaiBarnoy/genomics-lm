from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as skm
import yaml


def load_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text()) or {}


def ensure_dir(p: str | Path) -> Path:
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: str | Path, data: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True))


def _one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((y.shape[0], n_classes), dtype=np.float32)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    metrics["accuracy"] = float(skm.accuracy_score(y_true, y_pred))
    try:
        metrics["macro_f1"] = float(skm.f1_score(y_true, y_pred, average="macro"))
    except Exception:
        pass
    # AUROC: try OVR multiclass if probabilities or decision scores provided
    if y_proba is not None:
        try:
            if y_proba.ndim == 1:
                # binary case needs probabilities for positive class
                metrics["auroc"] = float(skm.roc_auc_score(y_true, y_proba))
            else:
                n_classes = y_proba.shape[1]
                y_true_oh = _one_hot(y_true, n_classes)
                metrics["auroc"] = float(skm.roc_auc_score(y_true_oh, y_proba, multi_class="ovr"))
        except Exception:
            pass
    return metrics


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, out_path: str | Path, normalize: bool = True) -> None:
    labels = np.unique(np.concatenate([y_true, y_pred]))
    cm = skm.confusion_matrix(y_true, y_pred, labels=labels, normalize=("true" if normalize else None))
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_title("Confusion Matrix" + (" (normalized)" if normalize else ""))
    plt.tight_layout(); ensure_dir(Path(out_path).parent); plt.savefig(out_path); plt.close(fig)


def plot_calibration(y_true: np.ndarray, y_proba: np.ndarray, out_path: str | Path, n_bins: int = 10) -> None:
    # Supports binary or per-class reliability diagram (plots first class otherwise)
    if y_proba.ndim > 1 and y_proba.shape[1] > 1:
        scores = y_proba.max(axis=1)
        y_bin = (y_true == y_proba.argmax(axis=1)).astype(int)
    else:
        scores = y_proba.reshape(-1)
        y_bin = y_true
    prob_true, prob_pred = skm.calibration_curve(y_bin, scores, n_bins=n_bins)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(prob_pred, prob_true, marker="o", label="model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect")
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("True frequency")
    ax.set_title("Calibration curve"); ax.legend()
    plt.tight_layout(); ensure_dir(Path(out_path).parent); plt.savefig(out_path); plt.close(fig)


def save_npz(path: str | Path, **arrays) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **arrays)


def load_npz(path: str | Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as blob:
        return {k: blob[k] for k in blob.files}


@dataclass
class EmbeddingPack:
    X: np.ndarray
    y: Optional[np.ndarray] = None
    ids: Optional[List[str]] = None

