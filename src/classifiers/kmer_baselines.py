from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .probes import compute_metrics


def _kmer_analyzer(k: int):
    def analyzer(s: str) -> List[str]:
        s = s.strip().upper().replace("U", "T")
        toks = []
        if len(s) < k:
            return toks
        for i in range(0, len(s) - k + 1):
            toks.append(s[i : i + k])
        return toks
    return analyzer


@dataclass
class KmerResult:
    vectorizer: object
    model: object
    metrics: Dict[str, float]
    y_pred: np.ndarray
    y_proba: Optional[np.ndarray]


def fit_kmer_logreg(seqs: List[str], y: np.ndarray, k: int = 3, tfidf: bool = True, C: float = 1.0, max_iter: int = 2000) -> KmerResult:
    vec = TfidfVectorizer(analyzer=_kmer_analyzer(k), lowercase=False, use_idf=tfidf, norm="l2")
    X = vec.fit_transform(seqs)
    clf = LogisticRegression(C=C, max_iter=max_iter, n_jobs=-1, multi_class="auto")
    clf.fit(X, y)
    y_pred = clf.predict(X)
    y_proba = None
    try:
        y_proba = clf.predict_proba(X)
    except Exception:
        pass
    metrics = compute_metrics(y, y_pred, y_proba)
    return KmerResult(vec, clf, metrics, y_pred, y_proba)


def fit_kmer_svm(seqs: List[str], y: np.ndarray, k: int = 3, tfidf: bool = True, C: float = 1.0) -> KmerResult:
    vec = TfidfVectorizer(analyzer=_kmer_analyzer(k), lowercase=False, use_idf=tfidf, norm="l2")
    X = vec.fit_transform(seqs)
    clf = LinearSVC(C=C)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    y_proba = None
    try:
        y_proba = clf.decision_function(X)
    except Exception:
        pass
    metrics = compute_metrics(y, y_pred, y_proba)
    return KmerResult(vec, clf, metrics, y_pred, y_proba)


def fit_kmer_xgb(seqs: List[str], y: np.ndarray, k: int = 3, tfidf: bool = True, **xgb_kwargs) -> KmerResult:
    try:
        from xgboost import XGBClassifier
    except Exception as exc:
        raise RuntimeError("xgboost not installed; pip install xgboost") from exc
    vec = TfidfVectorizer(analyzer=_kmer_analyzer(k), lowercase=False, use_idf=tfidf, norm="l2")
    X = vec.fit_transform(seqs)
    clf = XGBClassifier(n_estimators=xgb_kwargs.get("n_estimators", 200), max_depth=xgb_kwargs.get("max_depth", 6), learning_rate=xgb_kwargs.get("learning_rate", 0.1), subsample=0.8, colsample_bytree=0.8, tree_method=xgb_kwargs.get("tree_method", "auto"))
    clf.fit(X, y)
    y_pred = clf.predict(X)
    y_proba = None
    try:
        y_proba = clf.predict_proba(X)
    except Exception:
        pass
    metrics = compute_metrics(y, y_pred, y_proba)
    return KmerResult(vec, clf, metrics, y_pred, y_proba)

