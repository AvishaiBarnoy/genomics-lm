#!/usr/bin/env python3
"""
Convert compressed dataset .npz files into raw uncompressed .npy arrays.
This enables true O(1) memory-mapping (mmap_mode="r") in PyTorch Datasets
without loading the entire dataset into RAM at startup.
"""
import argparse
import sys
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Convert .npz to uncompressed .npy for true memory-mapping")
    parser.add_argument("npz_path", type=str, help="Path to the source .npz file")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory (defaults to directory of npz)")
    args = parser.parse_args()

    npz_path = Path(args.npz_path)
    if not npz_path.exists():
        print(f"Error: {npz_path} does not exist.", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir) if args.out_dir else npz_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {npz_path}...")
    with np.load(npz_path, allow_pickle=False) as data:
        keys = list(data.keys())
        print(f"Found keys: {keys}")
        
        for key in keys:
            arr = data[key]
            out_name = npz_path.stem + f"_{key}.npy"
            out_path = out_dir / out_name
            print(f"Saving {key} (shape={arr.shape}, dtype={arr.dtype}) to {out_path}...")
            np.save(out_path, arr, allow_pickle=False, fix_imports=False)
            
    print("Done. True memory-mapping is now enabled for this dataset.")


if __name__ == "__main__":
    main()
