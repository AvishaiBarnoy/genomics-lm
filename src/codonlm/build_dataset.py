#!/usr/bin/env python3
"""
Group-aware split by genome to prevent leakage.
Creates train/val/test NPZ packs.

Args:
  --test_frac 0.1
  --val_frac 0.1
  --group_meta data/processed/cds_meta.tsv
"""

from pathlib import Path
import argparse, numpy as np, random, collections

def load_lines(path):
    with open(path) as f:
        for line in f: yield [int(x) for x in line.strip().split()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="data/processed/codon_ids.txt")
    ap.add_argument("--group_meta", default="data/processed/cds_meta.tsv")
    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument("--windows_per_seq", type=int, default=2)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--test_frac", type=float, default=0.1)
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    rng = random.Random(args.seed); np.random.seed(args.seed)

    # load sequences
    seqs = list(load_lines(args.ids))
    # load groups
    groups = []
    with open(args.group_meta) as f:
        next(f)
        for line in f:
            i,g = line.strip().split("\t")
            groups.append(g)
    assert len(groups)==len(seqs), "meta and ids must align"

    # split by unique groups (fallback to sequence-level split if too few groups)
    uniq = list(sorted(set(groups)))
    rng.shuffle(uniq)

    buckets = {"train": [], "val": [], "test": []}

    if len(uniq) < 3:
        # Fallback: not enough distinct genomes to form 3 splits.
        idx = list(range(len(seqs)))
        rng.shuffle(idx)
        n_total = len(idx)
        n_test_seq = max(1, int(n_total * args.test_frac)) if n_total > 2 else (1 if n_total > 0 else 0)
        n_val_seq = max(1, int((n_total - n_test_seq) * args.val_frac)) if (n_total - n_test_seq) > 1 else 0
        test_idx = set(idx[:n_test_seq])
        val_idx = set(idx[n_test_seq:n_test_seq + n_val_seq])
        for i, arr in enumerate(seqs):
            if i in test_idx:
                buckets["test"].append(arr)
            elif i in val_idx:
                buckets["val"].append(arr)
            else:
                buckets["train"].append(arr)
    else:
        n_test = max(1, int(len(uniq) * args.test_frac))
        n_val = max(1, int(len(uniq) * args.val_frac))
        # Ensure at least one group remains for training
        if n_test + n_val >= len(uniq):
            # reduce val first, then test if needed
            n_val = max(0, len(uniq) - 1 - n_test)
            if n_test + n_val >= len(uniq):
                n_test = max(0, len(uniq) - 1)

        test_groups = set(uniq[:n_test])
        val_groups = set(uniq[n_test:n_test + n_val])
        train_groups = set(uniq[n_test + n_val:])

        for arr, g in zip(seqs, groups):
            key = "train" if g in train_groups else "val" if g in val_groups else "test"
            buckets[key].append(arr)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    def pack(name, subset):
        Xs, Ys = [], []
        for arr in subset:
            if len(arr)<=2: continue
            for _ in range(args.windows_per_seq):
                if len(arr) <= args.block_size+1:
                    x = arr[:-1]; y = arr[1:]
                    pad = [0]*max(0, args.block_size-len(x))
                    x=(x+pad)[:args.block_size]; y=(y+pad)[:args.block_size]
                else:
                    i = rng.randrange(0, len(arr)-args.block_size-1)
                    x = arr[i:i+args.block_size]
                    y = arr[i+1:i+1+args.block_size]
                Xs.append(x); Ys.append(y)
        X = np.array(Xs, dtype=np.int32)
        Y = np.array(Ys, dtype=np.int32)
        out = Path(args.out_dir)/f"{name}_bs{args.block_size}.npz"
        np.savez_compressed(out, X=X, Y=Y)
        print(f"[build] {name}: {X.shape} → {out}")

    for name in ("train","val","test"):
        pack(name, buckets[name])

if __name__ == "__main__":
    main()
