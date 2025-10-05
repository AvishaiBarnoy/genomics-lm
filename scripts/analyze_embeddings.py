"""Embedding analysis for TinyGPT runs."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ._shared import ArtifactError, ensure_run_layout, load_artifacts, load_token_list

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
except Exception:  # pragma: no cover - optional dependency
    KMeans = None
    silhouette_score = None


def _compute_pca(embeddings: np.ndarray, k: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    mean = embeddings.mean(axis=0, keepdims=True)
    centered = embeddings - mean
    u, s, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[:k]
    transformed = centered @ components.T
    explained = (s[:k] ** 2) / np.sum(s ** 2)
    return transformed, explained


def _nearest_neighbors(embeddings: np.ndarray, tokens: list[str], top_k: int = 5) -> list[tuple]:
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    normed = embeddings / norm
    sims = normed @ normed.T
    rows = []
    for idx in range(embeddings.shape[0]):
        order = np.argsort(-sims[idx])
        neighbors = []
        for j in order:
            if j == idx:
                continue
            neighbors.append((tokens[j], float(sims[idx, j])))
            if len(neighbors) >= top_k:
                break
        rows.append((tokens[idx], neighbors))
    return rows


def _load_motif_overlay(run_dir: Path) -> Optional[dict]:
    motif_path = run_dir / "motif_clusters.npz"
    if not motif_path.exists():
        return None
    try:
        with np.load(motif_path, allow_pickle=True) as data:
            payload = {k: data[k] for k in data.files}
    except Exception as exc:  # pragma: no cover - best effort
        print(f"[emb] failed to load motif_clusters: {exc}")
        return None
    return payload


def main(argv: Optional[Iterable[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    args = ap.parse_args(argv)

    paths = ensure_run_layout(args.run_id)
    run_dir, charts_dir, tables_dir = paths["run"], paths["charts"], paths["tables"]

    try:
        artifacts = load_artifacts(args.run_id)
    except ArtifactError as exc:
        print(f"[emb] {exc}")
        return

    embeddings = artifacts.get("token_embeddings")
    if embeddings is None or embeddings.size == 0:
        print("[emb] token_embeddings missing; aborting")
        return

    tokens = load_token_list(run_dir)
    coords, explained = _compute_pca(embeddings, k=2)

    plt.figure(figsize=(6, 6))
    plt.scatter(coords[:, 0], coords[:, 1], alpha=0.7)
    for token, x, y in zip(tokens, coords[:, 0], coords[:, 1]):
        plt.text(x, y, token, fontsize=6, ha="center", va="center")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Token embedding PCA")
    plt.tight_layout()
    plt.savefig(charts_dir / "emb_pca.png", dpi=200)
    plt.close()

    neighbors = _nearest_neighbors(embeddings, tokens, top_k=5)
    nn_path = tables_dir / "nearest_neighbors.csv"
    with nn_path.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["token"] + [f"neighbor_{i}" for i in range(1, 6)] + [f"sim_{i}" for i in range(1, 6)]
        writer.writerow(header)
        for token, neigh in neighbors:
            row = [token]
            row.extend([n[0] for n in neigh] + [""] * (5 - len(neigh)))
            row.extend([f"{n[1]:.4f}" for n in neigh] + [""] * (5 - len(neigh)))
            writer.writerow(row)

    silhouette = None
    if KMeans is not None and silhouette_score is not None and embeddings.shape[0] >= 5:
        try:
            kmeans = KMeans(n_clusters=min(8, embeddings.shape[0] // 2), n_init="auto")
            labels = kmeans.fit_predict(embeddings)
            silhouette = float(silhouette_score(embeddings, labels))
        except Exception as exc:  # pragma: no cover
            print(f"[emb] silhouette computation failed: {exc}")
            silhouette = None
    else:
        if KMeans is None:
            print("[emb] scikit-learn not available; skipping silhouette score")

    quality_path = tables_dir / "embed_quality.txt"
    lines = [
        f"PCA explained variance (PC1, PC2): {explained[0]:.4f}, {explained[1]:.4f}",
    ]
    if silhouette is not None:
        lines.append(f"Silhouette score: {silhouette:.4f}")
    else:
        lines.append("Silhouette score: NA")
    quality_path.write_text("\n".join(lines) + "\n")

    overlay = _load_motif_overlay(run_dir)
    if overlay and {"embeddings", "labels"}.issubset(overlay.keys()):
        motif_emb = np.asarray(overlay["embeddings"])
        motif_labels = np.asarray(overlay["labels"]).astype(str)
        motif_coords, _ = _compute_pca(motif_emb, k=2)
        plt.figure(figsize=(6, 6))
        scatter = plt.scatter(motif_coords[:, 0], motif_coords[:, 1], c=motif_labels)
        plt.legend(*scatter.legend_elements(), title="Motif")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("Motif clusters (PCA)")
        plt.tight_layout()
        plt.savefig(charts_dir / "emb_pca_motifs.png", dpi=200)
        plt.close()

    print(f"[emb] wrote PCA plot and nearest neighbors to {run_dir}")


if __name__ == "__main__":
    main()

