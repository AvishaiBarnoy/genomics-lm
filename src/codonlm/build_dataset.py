#!/usr/bin/env python3
"""
Pack codon id lines into contiguous arrays and create train/val splits.
We store two .npz files with arrays X (inputs) and Y (targets), already cropped to block_size.

Why:
- Random windowing reduces overfitting on gene starts/ends.
- Pre-packing avoids tokenization cost during training.

Key params:
- --block_size: context length in codons (M2 8GB → 256; M4 16GB → 512)
- --train_frac: train/val split ratio
- --windows_per_seq: number of random crops per input sequence (data augmentation)
"""

from pathlib import Path
import argparse, numpy as np, random

def load_lines(path):
    with open(path) as f:
        for line in f:
            yield [int(x) for x in line.strip().split()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="data/processed/codon_ids.txt")
    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument("--train_frac", type=float, default=0.9)
    ap.add_argument("--windows_per_seq", type=int, default=2)
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    rng = random.Random(args.seed); np.random.seed(args.seed)

    seqs = list(load_lines(args.ids))
    rng.shuffle(seqs)
    n_train = int(len(seqs)*args.train_frac)
    splits = [("train", seqs[:n_train]), ("val", seqs[n_train:])]

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    for name, subset in splits:
        Xs, Ys = [], []
        for arr in subset:
            if len(arr) <= 2: 
                continue
            for _ in range(args.windows_per_seq):
                if len(arr) <= args.block_size+1:
                    x = arr[:-1]
                    y = arr[1:]
                    # pad if short
                    pad = [0]*max(0, args.block_size-len(x))
                    x = (x+pad)[:args.block_size]
                    y = (y+pad)[:args.block_size]
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

if __name__ == "__main__":
    main()

