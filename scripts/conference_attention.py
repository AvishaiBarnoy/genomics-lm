"""
Conference Figure 2 — Attention Head Specialization Heatmaps.

Loads attention tensors from artifacts.npz and produces:
  1. A head-specialization overview grid (layers × heads) showing each head's
     entropy, start-codon bias, and stop-codon bias.
  2. Individual attention heatmaps for "interesting" heads (lowest entropy /
     highest bias) showing the full (T×T) attention pattern on a representative
     sequence.
  3. A summary bar chart: "start-codon attention score" across all L×H heads,
     to identify which heads specialize on ATG boundaries.

Usage:
    python -m scripts.conference_attention 2026-06-15_stage2.6_10L8H_d384_e10
    python -m scripts.conference_attention --run_dir runs/2026-06-15_stage2.6_10L8H_d384_e10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def _entropy(attn_row: np.ndarray) -> float:
    """Shannon entropy of a single attention distribution (nats)."""
    p = attn_row + 1e-12
    return float(-np.sum(p * np.log(p)))


def _head_stats(
    attn: np.ndarray,  # (L, B, H, T, T)
    tokens: list[str],
    start_tok: str = "ATG",
    stop_toks: tuple = ("TAA", "TAG", "TGA"),
) -> np.ndarray:
    """
    Returns array of shape (L, H, 3):
      [0] avg entropy across positions (lower = more focused)
      [1] avg attention score on start codon columns
      [2] avg attention score on stop codon columns
    """
    L, B, H, T, _ = attn.shape
    tok_arr = np.array(tokens[:T])

    start_cols = np.where(tok_arr == start_tok)[0]
    stop_cols = np.where(np.isin(tok_arr, list(stop_toks)))[0]

    stats = np.zeros((L, H, 3))
    for l in range(L):
        for h in range(H):
            head = attn[l, 0, h]  # (T, T) — use first batch item
            # Avg entropy (mean over source positions)
            ents = [_entropy(head[i]) for i in range(T)]
            stats[l, h, 0] = np.mean(ents)
            # Avg attention mass on start columns
            if len(start_cols):
                stats[l, h, 1] = float(head[:, start_cols].mean())
            # Avg attention mass on stop columns
            if len(stop_cols):
                stats[l, h, 2] = float(head[:, stop_cols].mean())
    return stats


def _plot_overview_grid(
    stats: np.ndarray,  # (L, H, 3)
    out_path: Path,
    dpi: int = 300,
) -> None:
    """Panel plot: three heatmaps — entropy, start bias, stop bias — over L×H grid."""
    L, H = stats.shape[:2]

    fig, axes = plt.subplots(1, 3, figsize=(15, max(4, L * 0.7 + 1.5)), facecolor="#0d0d0d")
    titles = ["Avg Entropy ↓\n(Focused heads)", "ATG Attention Bias ↑\n(Start recognition)", "Stop Codon Bias ↑\n(Termination)"]
    cmaps = ["viridis_r", "magma", "magma"]
    metrics = [stats[:, :, 0], stats[:, :, 1], stats[:, :, 2]]

    for ax, title, cmap, metric in zip(axes, titles, cmaps, metrics):
        ax.set_facecolor("#0d0d0d")
        im = ax.imshow(metric, aspect="auto", cmap=cmap, interpolation="nearest")
        ax.set_xlabel("Head", color="white", fontsize=9)
        ax.set_ylabel("Layer", color="white", fontsize=9)
        ax.set_xticks(range(H))
        ax.set_xticklabels([f"H{i}" for i in range(H)], fontsize=7, color="white")
        ax.set_yticks(range(L))
        ax.set_yticklabels([f"L{i}" for i in range(L)], fontsize=7, color="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")
        ax.set_title(title, color="white", fontsize=9, pad=8)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color="white", labelsize=7)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

        # Annotate best cells
        best_l, best_h = np.unravel_index(
            np.argmax(metric) if "Entropy" not in title else np.argmin(metric),
            metric.shape
        )
        ax.add_patch(plt.Rectangle(
            (best_h - 0.5, best_l - 0.5), 1, 1,
            linewidth=2, edgecolor="#00d4aa", facecolor="none"
        ))

    fig.suptitle(
        "Attention Head Specialization Overview\n"
        "CodonLM — Stage 2.6 (10L·8H·d384)",
        color="white", fontsize=12, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[attn] Overview grid → {out_path}")


def _plot_head_heatmap(
    attn_matrix: np.ndarray,  # (T, T)
    tokens: list[str],
    title: str,
    out_path: Path,
    dpi: int = 300,
    max_tokens: int = 60,
) -> None:
    """Single attention head heatmap for a representative prefix of the sequence."""
    T = min(max_tokens, attn_matrix.shape[0])
    matrix = attn_matrix[:T, :T]
    tok_labels = [t if len(t) == 3 and t.isalpha() else t[:6] for t in tokens[:T]]

    fig, ax = plt.subplots(figsize=(max(8, T * 0.22), max(7, T * 0.20)), facecolor="#0d0d0d")
    ax.set_facecolor("#0d0d0d")

    im = ax.imshow(matrix, aspect="auto", cmap="inferno", vmin=0, vmax=matrix.max(), interpolation="nearest")

    ax.set_xticks(range(T))
    ax.set_xticklabels(tok_labels, rotation=90, fontsize=5.5, fontfamily="monospace", color="white")
    ax.set_yticks(range(T))
    ax.set_yticklabels(tok_labels, fontsize=5.5, fontfamily="monospace", color="white")

    # Highlight ATG and stop codon columns/rows
    for j, tok in enumerate(tokens[:T]):
        if tok == "ATG":
            ax.axvline(j, color="#00d4aa", linewidth=1.2, alpha=0.6)
            ax.axhline(j, color="#00d4aa", linewidth=1.2, alpha=0.6)
        elif tok in ("TAA", "TAG", "TGA"):
            ax.axvline(j, color="#ff4444", linewidth=1.2, alpha=0.6)
            ax.axhline(j, color="#ff4444", linewidth=1.2, alpha=0.6)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.ax.yaxis.set_tick_params(color="white", labelsize=7)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    cbar.set_label("Attention weight", color="white", fontsize=8)

    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")

    ax.set_title(title, color="white", fontsize=9, pad=8, fontweight="bold")
    ax.set_xlabel("Key (source) position", color="white", fontsize=8)
    ax.set_ylabel("Query (target) position", color="white", fontsize=8)

    # Legend
    from matplotlib.lines import Line2D
    legend_lines = [
        Line2D([0], [0], color="#00d4aa", linewidth=2, label="ATG — Start Codon"),
        Line2D([0], [0], color="#ff4444", linewidth=2, label="TAA/TAG/TGA — Stop Codons"),
    ]
    ax.legend(handles=legend_lines, loc="upper left", fontsize=7,
              framealpha=0.2, facecolor="#1a1a1a", edgecolor="#444444", labelcolor="white")

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[attn] Head heatmap → {out_path}")


def _plot_bias_barchart(
    stats: np.ndarray,  # (L, H, 3)
    out_path: Path,
    dpi: int = 300,
    top_k: int = 8,
) -> None:
    """Bar chart of start-codon and stop-codon attention bias for all L×H heads."""
    L, H = stats.shape[:2]
    head_labels = [f"L{l}·H{h}" for l in range(L) for h in range(H)]
    start_scores = stats[:, :, 1].ravel()
    stop_scores = stats[:, :, 2].ravel()

    # Sort by start bias
    order = np.argsort(-start_scores)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), facecolor="#0d0d0d")

    for ax, scores, label, color in zip(
        axes,
        [start_scores[order], stop_scores[order]],
        ["ATG (Start Codon) Attention Bias", "Stop Codon (TAA/TAG/TGA) Attention Bias"],
        ["#00d4aa", "#ff4444"]
    ):
        ax.set_facecolor("#0d0d0d")
        bars = ax.bar(
            range(len(head_labels)),
            scores,
            color=[color if i < top_k else "#444444" for i in range(len(head_labels))],
            alpha=0.85,
            edgecolor="none",
        )
        # Highlight top-K
        for i in range(top_k):
            bars[i].set_edgecolor("white")
            bars[i].set_linewidth(0.8)

        ax.set_xticks(range(len(head_labels)))
        ax.set_xticklabels(
            [head_labels[o] for o in order],
            rotation=60, fontsize=6, fontfamily="monospace", color="white"
        )
        ax.set_ylabel(label, color="white", fontsize=8)
        ax.tick_params(axis="y", labelcolor="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")
        ax.axhline(scores.mean(), color="white", linewidth=0.8, linestyle="--", alpha=0.4)
        ax.text(len(head_labels) - 1, scores.mean() * 1.02, "mean", color="#aaaaaa", fontsize=7, ha="right")

    fig.suptitle(
        "Head-Level Attention Bias Towards Functional Codons\n"
        "CodonLM — Stage 2.6  |  Cyan = top-8 most specialized heads",
        color="white", fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[attn] Bias bar chart → {out_path}")


def main(argv: Optional[Iterable[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Conference attention head specialization figures")
    ap.add_argument("run_id", nargs="?")
    ap.add_argument("--run_dir", help="Path to the run directory")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--max_tokens", type=int, default=60,
                    help="Max tokens to show on individual heatmaps (default: 60)")
    args = ap.parse_args(argv)

    # Resolve run directory
    if args.run_dir:
        run_dir = Path(args.run_dir)
        run_id = run_dir.name
    elif args.run_id:
        run_dir = Path("runs") / args.run_id
        run_id = args.run_id
    else:
        candidates = sorted(Path("runs").iterdir(), reverse=True)
        candidates = [c for c in candidates if c.is_dir() and (c / "artifacts.npz").exists()]
        if not candidates:
            print("[attn] No runs with artifacts.npz found.")
            sys.exit(1)
        run_dir = candidates[0]
        run_id = run_dir.name
        print(f"[attn] Auto-selected run: {run_id}")

    artifacts_path = run_dir / "artifacts.npz"
    itos_path = run_dir / "itos.txt"

    if not artifacts_path.exists():
        print(f"[attn] artifacts.npz not found in {run_dir}")
        sys.exit(1)

    data = np.load(artifacts_path, allow_pickle=True)
    attn: np.ndarray = data["attn"]     # (L, B, H, T, T)
    val_inputs: np.ndarray = data.get("val_inputs", None)
    tokens: list[str] = itos_path.read_text().splitlines() if itos_path.exists() else [str(i) for i in range(attn.shape[3])]

    L, B, H, T, _ = attn.shape
    print(f"[attn] Loaded attention: {L} layers × {B} batch × {H} heads × {T}×{T}")

    charts_dir = run_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    conf_dir = Path("conference/figures")
    conf_dir.mkdir(parents=True, exist_ok=True)

    # For input-aligned token labels, use first sequence tokens from val_inputs if available
    seq_tokens = tokens
    if val_inputs is not None:
        first_seq_ids = val_inputs[0].tolist()
        seq_tokens = [tokens[i] if i < len(tokens) else "?" for i in first_seq_ids]

    # -----------------------------------------------------------------------
    # 1. Compute head statistics
    # -----------------------------------------------------------------------
    print("[attn] Computing head specialization statistics...")
    stats = _head_stats(attn, seq_tokens)

    # -----------------------------------------------------------------------
    # 2. Overview grid
    # -----------------------------------------------------------------------
    overview_path = charts_dir / "conference_attn_overview.png"
    _plot_overview_grid(stats, overview_path, dpi=args.dpi)
    import shutil
    shutil.copy(overview_path, conf_dir / "fig2a_attn_overview.png")

    # -----------------------------------------------------------------------
    # 3. Individual heatmaps for top-3 most focused heads (lowest entropy)
    # -----------------------------------------------------------------------
    entropy = stats[:, :, 0]
    flat_order = np.argsort(entropy.ravel())  # ascending = most focused
    plotted = 0
    for rank, flat_idx in enumerate(flat_order[:6]):
        l, h = np.unravel_index(flat_idx, (L, H))
        matrix = attn[l, 0, h]  # (T, T) first sample
        fname = charts_dir / f"conference_attn_L{l}_H{h}_focused.png"
        title = (
            f"L{l}·H{h} — Most Focused Head (rank #{rank+1})\n"
            f"Entropy={entropy[l,h]:.3f} | ATG-bias={stats[l,h,1]:.4f} | Stop-bias={stats[l,h,2]:.4f}"
        )
        _plot_head_heatmap(matrix, seq_tokens, title, fname, dpi=args.dpi, max_tokens=args.max_tokens)
        if plotted < 2:
            shutil.copy(fname, conf_dir / f"fig2b_attn_head_focused_rank{rank+1}.png")
        plotted += 1
        if plotted >= 3:
            break

    # -----------------------------------------------------------------------
    # 4. Top ATG-bias head heatmap
    # -----------------------------------------------------------------------
    start_bias = stats[:, :, 1]
    best_l, best_h = np.unravel_index(np.argmax(start_bias), (L, H))
    matrix = attn[best_l, 0, best_h]
    fname = charts_dir / f"conference_attn_L{best_l}_H{best_h}_start_specialist.png"
    title = (
        f"L{best_l}·H{best_h} — Start-Codon Specialist\n"
        f"ATG-bias={start_bias[best_l,best_h]:.4f} (highest across all {L*H} heads)"
    )
    _plot_head_heatmap(matrix, seq_tokens, title, fname, dpi=args.dpi, max_tokens=args.max_tokens)
    shutil.copy(fname, conf_dir / "fig2c_attn_start_specialist.png")

    # -----------------------------------------------------------------------
    # 5. Bias bar chart across all heads
    # -----------------------------------------------------------------------
    bar_path = charts_dir / "conference_attn_bias_barchart.png"
    _plot_bias_barchart(stats, bar_path, dpi=args.dpi)
    shutil.copy(bar_path, conf_dir / "fig2d_attn_bias_barchart.png")

    print(f"\n[attn] ✅ All figures saved to {charts_dir} and {conf_dir}")
    print(f"[attn] Summary:")
    print(f"  Most focused head:    L{np.unravel_index(np.argmin(entropy), (L,H))[0]}·H{np.unravel_index(np.argmin(entropy), (L,H))[1]}  (entropy={entropy.min():.3f})")
    print(f"  Best ATG specialist:  L{best_l}·H{best_h}  (bias={start_bias[best_l,best_h]:.4f})")


if __name__ == "__main__":
    main()
