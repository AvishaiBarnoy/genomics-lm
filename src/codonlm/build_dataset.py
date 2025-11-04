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
        """
        Pack multiple CDS into windows up to block_size with <SEP> separators to prevent cross-ORF leakage.

        Policy:
          - Each CDS already includes <BOS_CDS> and <EOS_CDS> from tokenization.
          - Insert <SEP> (id=3) between CDS segments when space permits.
          - If a CDS overflows the remaining space, truncate and carry remainder to the next window.
        """
        SEP_ID = 3  # matches codon_tokenize.py SPECIALS order
        PAD_ID = 0

        # Prepare per-sequence offsets to resume when truncated
        seqs = [arr for arr in subset if len(arr) > 2]
        if not seqs:
            X = np.zeros((0, args.block_size), dtype=np.int32)
            Y = np.zeros((0, args.block_size), dtype=np.int32)
            out = Path(args.out_dir) / f"{name}_bs{args.block_size}.npz"
            np.savez_compressed(out, X=X, Y=Y)
            print(f"[build] {name}: {X.shape} → {out}")
            return

        # Number of windows: keep similar scale as before
        windows_goal = max(1, args.windows_per_seq * len(seqs))
        Xs, Ys = [], []

        # Create an index queue we can shuffle each pass
        indices = list(range(len(seqs)))
        offsets = [0] * len(seqs)
        cur_ptr = 0

        for _ in range(windows_goal):
            rng.shuffle(indices)
            buf: List[int] = []
            # Fill window up to block_size
            for idx in indices:
                if len(buf) >= args.block_size:
                    break
                arr = seqs[idx]
                off = offsets[idx]
                if off >= len(arr):
                    continue
                # How many tokens can we copy from this CDS
                room = args.block_size - len(buf)
                take = min(room, len(arr) - off)
                if take <= 0:
                    continue
                buf.extend(arr[off : off + take])
                offsets[idx] += take
                # If CDS ended and we still have room, place a SEP
                if offsets[idx] >= len(arr) and len(buf) < args.block_size:
                    buf.append(SEP_ID)
            # Build x/y with padding
            x = buf[:-1]
            y = buf[1:]
            # Ensure length
            if len(x) < args.block_size:
                pad_n = args.block_size - len(x)
                x = x + [PAD_ID] * pad_n
                y = y + [PAD_ID] * pad_n
            else:
                x = x[: args.block_size]
                y = y[: args.block_size]
            Xs.append(x)
            Ys.append(y)

        X = np.array(Xs, dtype=np.int32)
        Y = np.array(Ys, dtype=np.int32)
        out = Path(args.out_dir) / f"{name}_bs{args.block_size}.npz"
        np.savez_compressed(out, X=X, Y=Y)
        print(f"[build] {name}: {X.shape} → {out}")
        X = np.array(Xs, dtype=np.int32)
        Y = np.array(Ys, dtype=np.int32)
        out = Path(args.out_dir)/f"{name}_bs{args.block_size}.npz"
        np.savez_compressed(out, X=X, Y=Y)
        print(f"[build] {name}: {X.shape} → {out}")

    for name in ("train","val","test"):
        pack(name, buckets[name])

if __name__ == "__main__":
    main()
