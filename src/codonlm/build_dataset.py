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
import argparse
import numpy as np
import random
from typing import List

def load_lines(path):
    """Loads whitespace-separated integers from each line of a file."""
    with open(path) as f:
        for line in f:
            yield [int(x) for x in line.strip().split()]

def main():
    """Builds and splits datasets by genome groups into train/val/test NPZ files."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="data/processed/codon_ids.txt")
    ap.add_argument("--group_meta", default="data/processed/cds_meta.tsv")
    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument("--windows_per_seq", type=int, default=2)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--test_frac", type=float, default=0.1)
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--pack_mode", choices=["multi", "single", "dynamic"], default="multi",
                    help="'multi': pack multiple CDS per window with <SEP>; 'single': one CDS per window; 'dynamic': save raw variable-length lists of arrays")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

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
        # fallback to sequence-level split
        indices = list(range(len(seqs)))
        rng.shuffle(indices)
        n_test = max(1, int(len(seqs) * args.test_frac))
        n_val = max(1, int(len(seqs) * args.val_frac))

        test_idx = set(indices[:n_test])
        val_idx = set(indices[n_test:n_test + n_val])

        for i, arr in enumerate(seqs):
            key = "val" if i in val_idx else "test" if i in test_idx else "train"
            buckets[key].append(arr)
    else:
        n_test = max(1, int(len(uniq) * args.test_frac))
        n_val = max(1, int(len(uniq) * args.val_frac))
        # Ensure at least one group remains for training
        if n_test + n_val >= len(uniq):
            # reduce val first, then test if needed
            n_val = max(0, len(uniq) - 1 - n_test)
            if n_test + n_val >= len(uniq):
                n_test = max(0, len(uniq) - 1)

        val_groups = set(uniq[n_test:n_test + n_val])
        train_groups = set(uniq[n_test + n_val:])

        for arr, g in zip(seqs, groups):
            key = "train" if g in train_groups else "val" if g in val_groups else "test"
            buckets[key].append(arr)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    def pack_multi(name, subset):
        """
        Pack multiple CDS into windows up to block_size with <SEP> separators to prevent cross-ORF leakage.

        Policy:
          - Each CDS already includes <BOS_CDS> and <EOS_CDS> from tokenization.
          - Insert <SEP> (id=3) between CDS segments when space permits.
          - If a CDS overflows the remaining space, truncate and carry remainder to the next window.
        """
        print(f"[build] packing multi: {name} (subset size={len(subset)})")
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

        # Number of windows: keep similar scale as before but cap by available tokens
        windows_goal = max(1, args.windows_per_seq * len(seqs))
        Xs, Ys = [], []

        # Create an index queue we can shuffle each pass
        indices = list(range(len(seqs)))
        offsets = [0] * len(seqs)

        made = 0
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
            # If we couldn't place at least two tokens, stop to avoid empty/pad-only windows
            if len(buf) < 2:
                break
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
            made += 1

            # If total remaining unconsumed tokens across all sequences is tiny, stop early
            remaining = 0
            for i, arr in enumerate(seqs):
                r = max(0, len(arr) - offsets[i])
                remaining += r
            if remaining < 2:
                break

        X = np.array(Xs, dtype=np.int32)
        Y = np.array(Ys, dtype=np.int32)
        out = Path(args.out_dir) / f"{name}_bs{args.block_size}.npz"
        np.savez_compressed(out, X=X, Y=Y)
        # packing stats
        total_tokens = int(X.size)
        sep_pct = float((X == 3).sum()) / total_tokens if total_tokens else 0.0
        pad_pct = float((X == 0).sum()) / total_tokens if total_tokens else 0.0
        cds_per_win = (X == 3).sum(axis=1) + 1 if X.shape[0] else np.array([0])
        avg_cds = float(np.mean(cds_per_win)) if X.shape[0] else 0.0
        print(f"[build] {name}: {X.shape} (windows={made}) → {out}")
        print(f"[pack-stats] {name}: avg_cds_per_window={avg_cds:.2f} sep_pct={sep_pct:.3f} pad_pct={pad_pct:.3f}")

    def pack_single(name, subset):
        """Packs each sequence independently into windows (one CDS per window, padded or cropped)."""
        Xs, Ys = [], []
        for arr in subset:
            if len(arr) <= 2:
                continue
            for _ in range(args.windows_per_seq):
                if len(arr) <= args.block_size + 1:
                    x = arr[:-1]
                    y = arr[1:]
                    pad = [0] * max(0, args.block_size - len(x))
                    x = (x + pad)[: args.block_size]
                    y = (y + pad)[: args.block_size]
                else:
                    i = rng.randrange(0, len(arr) - args.block_size - 1)
                    x = arr[i : i + args.block_size]
                    y = arr[i + 1 : i + 1 + args.block_size]
                Xs.append(x)
                Ys.append(y)
        X = np.array(Xs, dtype=np.int32)
        Y = np.array(Ys, dtype=np.int32)
        out = Path(args.out_dir) / f"{name}_bs{args.block_size}.npz"
        np.savez_compressed(out, X=X, Y=Y)
        print(f"[build] {name}: {X.shape} → {out}")

    def pack_dynamic(name, subset):
        """Packs each sequence independently without padding. Concatenates them into a flat array and saves lengths to avoid pickle."""
        filtered = []
        for arr in subset:
            if len(arr) <= 2:
                continue
            if len(arr) > args.block_size:
                arr = arr[-args.block_size:]
            filtered.append(arr)

        if not filtered:
            flat_X = np.zeros((0,), dtype=np.int32)
            lengths = np.zeros((0,), dtype=np.int32)
        else:
            flat_X = np.concatenate([np.array(x, dtype=np.int32) for x in filtered])
            lengths = np.array([len(x) for x in filtered], dtype=np.int32)

        out = Path(args.out_dir) / f"{name}_bs{args.block_size}.npz"
        np.savez_compressed(out, X=flat_X, lengths=lengths)
        print(f"[build] {name} (dynamic): {len(filtered)} sequences → {out}")

    for name in ("train", "val", "test"):
        if args.pack_mode == "single":
            pack_single(name, buckets[name])
        elif args.pack_mode == "dynamic":
            pack_dynamic(name, buckets[name])
        else:
            pack_multi(name, buckets[name])

if __name__ == "__main__":
    main()
