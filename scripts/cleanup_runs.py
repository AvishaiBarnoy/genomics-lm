#!/usr/bin/env python3
"""
scripts/cleanup_runs.py — Local Run/Checkpoint Clean-up Utility

This script scans the `runs/` directory for untracked checkpoints and run directories.
It allows listing size footprints and safely purging intermediate weights (e.g. last.pt,
epoch_*.pt) or old runs to reclaim local disk space.
"""
import argparse
from pathlib import Path
import shutil
import sys
import time

# Keep important/final checkpoint files by default
PROTECTED_FILENAMES = {
    "best.pt",
    "best_critic.pt",
    "best_ebm.pt",
    "structural_regression_probes.pt",
}

REPO_DIR = Path(__file__).resolve().parents[1]


def get_dir_size_mb(path: Path) -> float:
    total = sum(f.stat().st_size for f in path.glob("**/*") if f.is_file())
    return total / (1024 * 1024)


def main():
    ap = argparse.ArgumentParser(description="Clean up intermediate local checkpoints and old runs")
    ap.add_argument(
        "--dry_run",
        action="store_true",
        default=False,
        help="List files/folders targeted for cleanup without deleting them."
    )
    ap.add_argument(
        "--keep_only_best",
        action="store_true",
        default=False,
        help="Delete all intermediate checkpoints (e.g., last.pt, epoch_*.pt) keeping only 'best' weights."
    )
    ap.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Skip interactive confirmation prompts."
    )
    args = ap.parse_args()

    # If no cleanup action is specified, default to dry-run reporting
    is_dry_run = args.dry_run or not args.keep_only_best

    runs_dir = REPO_DIR / "runs"
    
    if not runs_dir.exists():
        print(f"No runs directory found at {runs_dir}.")
        return

    print("=== Scanning runs directory ===")
    run_folders = [p for p in runs_dir.iterdir() if p.is_dir() and p.name != "_summary"]
    print(f"Found {len(run_folders)} run directories.")
    
    total_reclaimed = 0.0

    # 1. Purge intermediate checkpoints
    if args.keep_only_best:
        print("\nScanning for intermediate checkpoint weights...")
        for folder in run_folders:
            chk_dir = folder / "checkpoints"
            target_dirs = [chk_dir, folder] # Checkpoints can be in root run dir or checkpoints subfolder
            
            for d in target_dirs:
                if not d.exists():
                    continue
                for item in d.glob("*.pt"):
                    if item.is_file() and item.name not in PROTECTED_FILENAMES:
                        size_mb = item.stat().st_size / (1024 * 1024)
                        total_reclaimed += size_mb
                        print(f"Target: [Intermediate Weight] {item.relative_to(REPO_DIR)} ({size_mb:.2f} MB)")
                        
                        if not is_dry_run:
                            if args.force or input(f"Confirm delete {item.name}? [y/N] ").lower() == "y":
                                item.unlink()
                                print(f"Deleted {item.name}")

    print("\n=== Cleanup Summary ===")
    if is_dry_run:
        print(f"Dry-run finished. Total space targetable for cleanup: {total_reclaimed:.2f} MB")
        if not args.keep_only_best:
            print("\nTo perform cleanup, specify an action:")
            print("  --keep_only_best    : Delete non-best model checkpoints.")
            print("Use --force to skip confirmations.")
    else:
        print(f"Cleanup finished. Reclaimed: {total_reclaimed:.2f} MB")


if __name__ == "__main__":
    main()
