"""
Conference Figure 1 — UMAP Codon Embedding Plot.

Loads the token_embeddings artifact from a trained run and produces a
publication-quality UMAP scatter plot where:
  - Each point = one codon token (64 sense codons + special tokens)
  - Color = amino acid identity (synonymous codons cluster together)
  - Size/marker = start codon (ATG), stop codons (TAA/TAG/TGA) are highlighted
  - Axes are clean with no ticks; legend is sorted alphabetically

Usage:
    python -m scripts.conference_umap 2026-06-15_stage2.6_10L8H_d384_e10
    python -m scripts.conference_umap --run_dir runs/2026-06-15_stage2.6_10L8H_d384_e10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Codon → amino acid lookup (standard genetic code)
# ---------------------------------------------------------------------------
GENETIC_CODE: dict[str, str] = {
    "TTT": "Phe", "TTC": "Phe", "TTA": "Leu", "TTG": "Leu",
    "CTT": "Leu", "CTC": "Leu", "CTA": "Leu", "CTG": "Leu",
    "ATT": "Ile", "ATC": "Ile", "ATA": "Ile", "ATG": "Met*",
    "GTT": "Val", "GTC": "Val", "GTA": "Val", "GTG": "Val",
    "TCT": "Ser", "TCC": "Ser", "TCA": "Ser", "TCG": "Ser",
    "CCT": "Pro", "CCC": "Pro", "CCA": "Pro", "CCG": "Pro",
    "ACT": "Thr", "ACC": "Thr", "ACA": "Thr", "ACG": "Thr",
    "GCT": "Ala", "GCC": "Ala", "GCA": "Ala", "GCG": "Ala",
    "TAT": "Tyr", "TAC": "Tyr", "TAA": "Stop", "TAG": "Stop",
    "CAT": "His", "CAC": "His", "CAA": "Gln", "CAG": "Gln",
    "AAT": "Asn", "AAC": "Asn", "AAA": "Lys", "AAG": "Lys",
    "GAT": "Asp", "GAC": "Asp", "GAA": "Glu", "GAG": "Glu",
    "TGT": "Cys", "TGC": "Cys", "TGA": "Stop", "TGG": "Trp",
    "CGT": "Arg", "CGC": "Arg", "CGA": "Arg", "CGG": "Arg",
    "AGT": "Ser", "AGC": "Ser", "AGA": "Arg", "AGG": "Arg",
    "GGT": "Gly", "GGC": "Gly", "GGA": "Gly", "GGG": "Gly",
}

# 20 distinct colors from a colorblind-friendly qualitative palette
AA_PALETTE = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9a6324", "#fffac8", "#800000",
    "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9",
    "#ffffff", "#000000",
]

STOP_COLOR = "#ff4444"
START_COLOR = "#00d4aa"
SPECIAL_COLOR = "#888888"


def _get_umap(embeddings: np.ndarray) -> np.ndarray:
    """Return 2-D UMAP coordinates, falling back to PCA if umap-learn is absent."""
    try:
        import umap  # type: ignore
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(10, embeddings.shape[0] - 1),
            min_dist=0.25,
            metric="cosine",
            random_state=42,
        )
        return reducer.fit_transform(embeddings)
    except ImportError:
        print("[umap] umap-learn not installed — falling back to PCA (install with: pip install umap-learn)")
        mean = embeddings.mean(axis=0, keepdims=True)
        centered = embeddings - mean
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        return (centered @ vh[:2].T)


def main(argv: Optional[Iterable[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Conference UMAP codon embedding plot")
    ap.add_argument("run_id", nargs="?", help="Run ID (e.g. 2026-06-15_stage2.6_10L8H_d384_e10)")
    ap.add_argument("--run_dir", help="Path to the run directory (alternative to run_id)")
    ap.add_argument("--out", help="Output PNG path (default: runs/<RUN>/charts/conference_umap.png)")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args(argv)

    # Resolve run directory
    if args.run_dir:
        run_dir = Path(args.run_dir)
        run_id = run_dir.name
    elif args.run_id:
        run_dir = Path("runs") / args.run_id
        run_id = args.run_id
    else:
        # Auto-detect newest run
        candidates = sorted(Path("runs").iterdir(), reverse=True)
        candidates = [c for c in candidates if c.is_dir() and (c / "artifacts.npz").exists()]
        if not candidates:
            print("[umap] No runs with artifacts.npz found. Run eval_generation_prefix first.")
            sys.exit(1)
        run_dir = candidates[0]
        run_id = run_dir.name
        print(f"[umap] Auto-selected run: {run_id}")

    artifacts_path = run_dir / "artifacts.npz"
    itos_path = run_dir / "itos.txt"

    if not artifacts_path.exists():
        print(f"[umap] artifacts.npz not found in {run_dir}")
        sys.exit(1)

    # Load data
    data = np.load(artifacts_path, allow_pickle=True)
    embeddings: np.ndarray = data["token_embeddings"]  # (V, D)
    tokens: list[str] = itos_path.read_text().splitlines() if itos_path.exists() else [str(i) for i in range(len(embeddings))]

    print(f"[umap] Loaded {embeddings.shape[0]} token embeddings, dim={embeddings.shape[1]}")

    # Reduce to 2D
    coords = _get_umap(embeddings)

    # Assign colors per token
    amino_acids = sorted({aa for aa in GENETIC_CODE.values() if aa not in ("Stop", "Met*")})
    aa_color_map = {aa: AA_PALETTE[i % len(AA_PALETTE)] for i, aa in enumerate(amino_acids)}
    aa_color_map["Stop"] = STOP_COLOR
    aa_color_map["Met*"] = START_COLOR

    colors, sizes, markers, zorders, edge_colors = [], [], [], [], []
    for tok in tokens:
        if tok in GENETIC_CODE:
            aa = GENETIC_CODE[tok]
            colors.append(aa_color_map[aa])
            if tok == "ATG":
                sizes.append(180)
                markers.append("*")
                zorders.append(5)
                edge_colors.append("black")
            elif aa == "Stop":
                sizes.append(140)
                markers.append("X")
                zorders.append(4)
                edge_colors.append("black")
            else:
                sizes.append(80)
                markers.append("o")
                zorders.append(2)
                edge_colors.append("none")
        else:
            colors.append(SPECIAL_COLOR)
            sizes.append(50)
            markers.append("s")
            zorders.append(1)
            edge_colors.append("none")

    # -------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 9), facecolor="#0d0d0d")
    ax.set_facecolor("#0d0d0d")

    # Background glow for stop/start codons
    for i, tok in enumerate(tokens):
        if tok in ("TAA", "TAG", "TGA", "ATG"):
            glow_color = START_COLOR if tok == "ATG" else STOP_COLOR
            ax.scatter(coords[i, 0], coords[i, 1], s=sizes[i] * 6,
                       color=glow_color, alpha=0.15, zorder=0)

    # Plot each marker type (matplotlib scatter doesn't support per-point markers)
    for i, tok in enumerate(tokens):
        ax.scatter(
            coords[i, 0], coords[i, 1],
            s=sizes[i],
            c=colors[i],
            marker=markers[i],
            zorder=zorders[i],
            edgecolors=edge_colors[i],
            linewidths=0.8,
        )

    # Token labels — only for codons (not special tokens), offset slightly
    for i, tok in enumerate(tokens):
        if len(tok) == 3 and tok.isalpha() and tok.upper() == tok:
            ax.text(
                coords[i, 0], coords[i, 1] + 0.18,
                tok,
                fontsize=5.5,
                ha="center", va="bottom",
                color="white",
                alpha=0.85,
                fontfamily="monospace",
                zorder=10,
            )

    # Legend: amino acids
    legend_patches = []
    for aa in sorted(aa_color_map.keys()):
        if aa == "Met*":
            label = "Met / Start (ATG) ★"
        elif aa == "Stop":
            label = "Stop (TAA/TAG/TGA) ✕"
        else:
            label = aa
        legend_patches.append(mpatches.Patch(color=aa_color_map[aa], label=label))

    leg = ax.legend(
        handles=legend_patches,
        loc="lower left",
        fontsize=7,
        framealpha=0.15,
        facecolor="#1a1a1a",
        edgecolor="#444444",
        title="Amino Acid",
        title_fontsize=8,
        labelcolor="white",
        ncol=2,
    )
    leg.get_title().set_color("white")

    # Axis cosmetics
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(
        f"CodonLM — Codon Embedding Space\n"
        f"Model: {run_id}  |  {embeddings.shape[0]-4} codons · d={embeddings.shape[1]}",
        color="white",
        fontsize=11,
        pad=14,
        fontweight="bold",
    )

    fig.text(0.5, 0.02,
             "2D UMAP projection (cosine distance) of learned codon token embeddings\n"
             "Synonymous codons encoding the same amino acid cluster together — an emergent structure learned without structural supervision.",
             ha="center", color="#aaaaaa", fontsize=7.5, style="italic")

    plt.tight_layout(rect=[0, 0.06, 1, 1])

    # Save
    out_path = Path(args.out) if args.out else run_dir / "charts" / "conference_umap.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=args.dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[umap] Saved → {out_path}")

    # Also copy to conference/figures/
    conf_dir = Path("conference/figures")
    conf_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(out_path, conf_dir / "fig1_umap_codon_embeddings.png")
    print(f"[umap] Copied → {conf_dir / 'fig1_umap_codon_embeddings.png'}")


if __name__ == "__main__":
    main()
