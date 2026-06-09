#!/usr/bin/env python3
"""
CLI entry point for mining motifs from a trained model.
Extracts hidden-state embeddings, clusters them, and saves the results.
"""

import argparse
import torch
import numpy as np
from src.eval.motif_extractor import MotifExtractor
from src.eval.motif_clusterer import MotifClusterer
from scripts._shared import load_model, ensure_run_layout, resolve_run, load_token_list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True, help="Run ID to analyze")
    ap.add_argument(
        "--data_npz", required=True, help="Path to data npz containing 'X' (token IDs)"
    )
    ap.add_argument(
        "--n_samples", type=int, default=100, help="Number of sequences to process"
    )
    ap.add_argument(
        "--window_size", type=int, default=9, help="Sliding window size (in codons)"
    )
    ap.add_argument("--stride", type=int, default=3, help="Sliding window stride")
    ap.add_argument(
        "--n_clusters",
        type=int,
        default=10,
        help="Number of clusters (or min_cluster_size for HDBSCAN)",
    )
    ap.add_argument(
        "--method",
        default="kmeans",
        choices=["kmeans", "hdbscan"],
        help="Clustering algorithm",
    )
    ap.add_argument(
        "--pca", type=int, default=50, help="Number of PCA components before clustering"
    )
    ap.add_argument(
        "--layer", type=int, default=-1, help="Layer index to extract from (-1 is last)"
    )
    args = ap.parse_args()

    run_id, run_dir = resolve_run(run_id=args.run_id)
    ensure_run_layout(run_id)
    out_dir = run_dir / "motif_mining"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[*] Loading model from {run_dir}...")
    model, spec = load_model(run_dir, device=device)

    # Load data
    print(f"[*] Loading data from {args.data_npz}...")
    data = np.load(args.data_npz)
    X_raw = data["X"][: args.n_samples]

    # Trim sequences to model's block_size if necessary
    if X_raw.shape[1] > spec.block_size:
        print(f"[!] Trimming sequences from {X_raw.shape[1]} to {spec.block_size}")
        X_raw = X_raw[:, : spec.block_size]

    X = torch.from_numpy(X_raw).long().to(device)

    # Extract embeddings
    extractor = MotifExtractor(model, window_size=args.window_size, stride=args.stride)
    print(
        f"[*] Extracting embeddings for {len(X)} sequences (window={args.window_size}, stride={args.stride})..."
    )

    # Identify special token IDs to exclude from windows
    tokens = load_token_list(run_dir)
    tok_stoi = {t: i for i, t in enumerate(tokens)}
    special_ids = [
        tok_stoi[s]
        for s in ["<pad>", "<BOS_CDS>", "<EOS_CDS>", "<SEP>"]
        if s in tok_stoi
    ]
    # Also check legacy/aliased specials
    for s in ["<bos>", "<eos>", "<eog>", "<PAD>"]:
        if s in tok_stoi:
            special_ids.append(tok_stoi[s])

    all_embeddings = []
    all_windows_tokens = []

    # Process one by one to avoid OOM on large hidden state stacks
    for i in range(len(X)):
        # extract returns (embeddings, kept_metadata)
        emb, metadata = extractor.extract(
            X[i : i + 1], layer_idx=args.layer, exclude_ids=special_ids
        )
        if len(emb) > 0:
            all_embeddings.append(emb.cpu().numpy())
            for _, start, end in metadata:
                all_windows_tokens.append([tokens[t] for t in X_raw[i, start:end]])

    if not all_embeddings:
        print(
            "[!] No windows kept after filtering special tokens. Try a smaller window or check data."
        )
        return

    X_emb = np.vstack(all_embeddings)
    print(f"[*] Extracted {len(X_emb)} clean windows (filtered out special tokens).")

    # Clustering
    print(
        f"[*] Clustering using {args.method} (n_clusters={args.n_clusters}, pca={args.pca})..."
    )
    clusterer = MotifClusterer(
        method=args.method, n_clusters=args.n_clusters, pca_components=args.pca
    )
    labels = clusterer.fit_predict(X_emb)
    centers = clusterer.get_centers(X_emb)

    # Motif Analysis & Report Generation
    print("[*] Generating motif report...")
    from src.eval.motif_analysis import calculate_pwm, get_consensus

    # Group windows by cluster
    clusters = {}
    for idx, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(all_windows_tokens[idx])

    # Generate Markdown report
    report = [f"# Motif Mining Report: {run_id}", ""]
    report.append(f"- **Sequences sampled:** {len(X_raw)}")
    report.append(f"- **Total windows:** {len(X_emb)}")
    report.append(f"- **Method:** {args.method}")
    report.append(f"- **Layer:** {args.layer}")
    report.append("")

    # Sort clusters by size (descending)
    sorted_labels = sorted(
        clusters.keys(), key=lambda c_key: len(clusters[c_key]), reverse=True
    )

    for label in sorted_labels:
        if label == -1:
            name = "Noise (HDBSCAN)"
        else:
            name = f"Cluster {label}"

        seqs = clusters[label]
        # Only compute PWM for the 64 codons to keep it clean (skip specials if possible)
        # But for now, use all tokens to be safe
        pwm = calculate_pwm(seqs, tokens)
        cons = get_consensus(pwm)

        report.append(f"## {name} (size={len(seqs)})")
        report.append(f"**Consensus:** `{cons}`")
        report.append("")
        report.append("**Top 5 Examples:**")
        for ex in seqs[:5]:
            report.append(f"- `{' '.join(ex)}`")
        report.append("")

    report_path = out_dir / "motif_report.md"
    report_path.write_text("\n".join(report))

    # Save results
    out_path = out_dir / "clusters.npz"
    np.savez(
        out_path,
        labels=labels,
        centers=centers,
        window_size=args.window_size,
        stride=args.stride,
        layer=args.layer,
        method=args.method,
    )

    print(f"[success] Motif mining complete. Results saved to {out_path}")


if __name__ == "__main__":
    main()
