"""Visualize attention maps for a collected run."""
from __future__ import annotations

import argparse
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ._shared import ArtifactError, ensure_run_layout, load_artifacts, load_token_list


def _plot_attention(attn: np.ndarray, tokens: list[str], out_path) -> None:
    plt.figure(figsize=(6, 5))
    plt.imshow(attn, aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar(label="Attention weight")
    n = min(len(tokens), attn.shape[0])
    plt.xticks(range(n), tokens[:n], rotation=90, fontsize=6)
    plt.yticks(range(n), tokens[:n], fontsize=6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main(argv: Optional[Iterable[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    args = ap.parse_args(argv)

    paths = ensure_run_layout(args.run_id)
    run_dir, charts_dir = paths["run"], paths["charts"]

    try:
        artifacts = load_artifacts(args.run_id)
    except ArtifactError as exc:
        print(f"[attn] {exc}")
        return

    attn = artifacts.get("attn")
    if attn is None or attn.size == 0:
        print("[attn] no attention tensors captured; skipping")
        return

    tokens = load_token_list(run_dir)
    # attn shape: (layers, batch, heads, T, T)
    layers = attn.shape[0]
    heads = attn.shape[2]
    T = attn.shape[3]

    for layer in range(layers):
        for head in range(heads):
            # only visualize the first sample for compactness
            matrix = attn[layer, 0, head]
            out = charts_dir / f"attn_L{layer}_H{head}_0-{T}.png"
            _plot_attention(matrix, tokens[:T], out)

    print(f"[attn] saved {layers * heads} attention heatmaps to {charts_dir}")


if __name__ == "__main__":
    main()

