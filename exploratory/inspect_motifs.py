#!/usr/bin/env python3
import argparse, os, numpy as np, pandas as pd
from collections import defaultdict, Counter

SPECIALS = ["<pad>", "<bos>", "<eog>", "<unk>", "<eos>"]
CODONS = [a+b+c for a in "ACGT" for b in "ACGT" for c in "ACGT"]
VOCAB = SPECIALS + CODONS
itos = {i:t for i,t in enumerate(VOCAB)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters_npz", required=True, help="outputs/motif_clusters.npz")
    ap.add_argument("--train_npz",    required=True, help="data/processed/train_bs256.npz")
    ap.add_argument("--k", type=int, default=9)
    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument("--samples", type=int, default=20000, help="should match mine_motifs run")
    ap.add_argument("--top", type=int, default=5, help="top example windows to list per cluster")
    ap.add_argument("--outdir", default="analysis/motifs_out")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    data = np.load(args.clusters_npz)
    centers = data["centers"]              # (C, d)
    labels  = data["labels"].astype(int)   # (N_windows,)
    C = centers.shape[0]

    # Reconstruct mapping from label index -> (seq_i, start_t)
    X = np.asarray(np.load(args.train_npz)["X"])  # (N, T)
    n_seq = min(args.samples, X.shape[0])
    mapping = []   # list of (i, t) for each window in the same order as labels
    for i in range(n_seq):
        T = min(args.block_size, X.shape[1])      # sequences were cropped to block_size when mining
        win = max(0, T - args.k + 1)
        for t in range(win):
            mapping.append((i, t))
    assert len(mapping) == len(labels), f"Window count mismatch: mapping={len(mapping)} labels={len(labels)}"

    # Cluster sizes
    sizes = np.bincount(labels, minlength=C)
    pd.DataFrame({"cluster": np.arange(C), "size": sizes}).to_csv(
        os.path.join(args.outdir, "cluster_sizes.csv"), index=False
    )

    # Build per-cluster example windows (as codon strings)
    # NOTE: X holds token IDs; map with itos. Windows are length k.
    per_cluster_examples = defaultdict(list)
    for idx, lab in enumerate(labels):
        if len(per_cluster_examples[lab]) >= args.top:  # keep a few examples per cluster
            continue
        i, t = mapping[idx]
        ids = X[i, t:t+args.k]
        toks = [itos.get(int(z), f"<{int(z)}>") for z in ids]
        per_cluster_examples[lab].append(" ".join(toks))

    # Per-cluster consensus (majority codon per position)
    consensus_rows = []
    for c in range(C):
        positions = [Counter() for _ in range(args.k)]
        # iterate all windows in cluster c
        for idx, lab in enumerate(labels):
            if lab != c: continue
            i, t = mapping[idx]
            ids = X[i, t:t+args.k]
            for p, tok_id in enumerate(ids):
                positions[p][itos.get(int(tok_id), "?")] += 1
        # majority call per position
        cons = [pos.most_common(1)[0][0] if pos else "-" for pos in positions]
        consensus_rows.append({"cluster": c, "size": int(sizes[c]), "consensus": " ".join(cons)})

    pd.DataFrame(consensus_rows).sort_values("size", ascending=False).to_csv(
        os.path.join(args.outdir, "cluster_consensus.csv"), index=False
    )

    # Write examples
    with open(os.path.join(args.outdir, "cluster_examples.txt"), "w") as f:
        for c in np.argsort(-sizes):
            print(f"[cluster {c}] size={sizes[c]}", file=f)
            for ex in per_cluster_examples[c]:
                print("  ", ex, file=f)

    print("[save]", args.outdir, "(cluster_sizes.csv, cluster_consensus.csv, cluster_examples.txt)")

if __name__ == "__main__":
    main()

