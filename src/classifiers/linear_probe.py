from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from .probes import compute_metrics


@dataclass
class ProbeResult:
    model: object
    metrics: Dict[str, float]
    y_pred: np.ndarray
    y_proba: Optional[np.ndarray]


def fit_logreg(X: np.ndarray, y: np.ndarray, C: float = 1.0, max_iter: int = 2000, multi_class: str = "auto") -> ProbeResult:
    clf = Pipeline([
        ("scaler", StandardScaler(with_mean=True)),
        ("clf", LogisticRegression(C=C, max_iter=max_iter, n_jobs=-1, multi_class=multi_class))
    ])
    clf.fit(X, y)
    y_pred = clf.predict(X)
    y_proba = None
    try:
        y_proba = clf.predict_proba(X)
    except Exception:
        pass
    metrics = compute_metrics(y, y_pred, y_proba)
    return ProbeResult(clf, metrics, y_pred, y_proba)


def fit_linear_svm(X: np.ndarray, y: np.ndarray, C: float = 1.0) -> ProbeResult:
    clf = Pipeline([
        ("scaler", StandardScaler(with_mean=True)),
        ("clf", LinearSVC(C=C))
    ])
    clf.fit(X, y)
    y_pred = clf.predict(X)
    y_proba = None
    # decision_function is not a probability; we can pass it for AUROC computation in OVR
    try:
        dec = clf.decision_function(X)
        y_proba = dec if isinstance(dec, np.ndarray) else None
    except Exception:
        pass
    metrics = compute_metrics(y, y_pred, y_proba)
    return ProbeResult(clf, metrics, y_pred, y_proba)

