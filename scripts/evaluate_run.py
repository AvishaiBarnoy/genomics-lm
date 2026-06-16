#!/usr/bin/env python3
"""
Orchestrates and runs standard evaluation tracks for genomics-lm models.
Modes:
  --mode quick  : Run test perplexity and basic sanity KPIs.
  --mode medium : Run quick + DNA structural awareness + stop codon placement.
  --mode full   : Run medium + DMS mutational zero-shot + gene essentiality benchmarks + SOTA report.

Usage:
  python -m scripts.evaluate_run --run_id <RUN_ID> --mode {quick,medium,full} [--device {cpu,mps,cuda}]
"""

import argparse
import sys
import subprocess
import os
from pathlib import Path

def run_cmd(cmd, env=None):
    print(f"\n[*] Executing: {' '.join(cmd)}")
    current_env = os.environ.copy()
    if env:
        current_env.update(env)
    try:
        subprocess.run(cmd, check=True, env=current_env)
        return True
    except subprocess.CalledProcessError as err:
        print(f"[!] Command failed: {' '.join(cmd)}")
        print(f"[!] Error code: {err.returncode}")
        return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True, help="Run ID of the model to evaluate")
    ap.add_argument("--mode", choices=["quick", "medium", "full"], default="quick", help="Evaluation depth")
    ap.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu", help="Device to run benchmarks on (CPU recommended to avoid MPS lockups)")
    ap.add_argument("--max_genes", type=int, default=10, help="Max genes to evaluate in prefix generation (medium/full mode)")
    ap.add_argument("--samples", type=int, default=2, help="Samples per prefix in prefix generation (medium/full mode)")
    args = ap.parse_args()

    run_id = args.run_id
    mode = args.mode
    device = args.device

    print(f"===========================================================")
    print(f"Starting Unified Evaluation Suite for: {run_id}")
    print(f"Mode: {mode.upper()} | Device: {device.upper()}")
    print(f"===========================================================")

    # Ensure python paths/etc
    python_bin = sys.executable

    # 1. Quick Evaluation (Loss, Perplexity, Sanity KPIs)
    print("\n>>> Phase 1: Quick Perplexity & Sanity KPIs <<<")

    # Run test perplexity
    cmd_ppl = [
        python_bin, "-m", "scripts.evaluate_test",
        "--run_id", run_id,
        "--data_dir", "data/processed/stage2.5_master_pack_v2"
    ]
    # Set FORCE_CPU if device is CPU
    env = {"FORCE_CPU": "1"} if device == "cpu" else {}
    if not run_cmd(cmd_ppl, env=env):
        sys.exit(1)

    # Run sanity KPIs
    cmd_kpis = [
        python_bin, "-m", "scripts.sanity_kpis",
        "--run_id", run_id,
        "--test_npz", "data/processed/stage2.5_master_pack_v2/test_bs512.npz"
    ]
    if not run_cmd(cmd_kpis, env=env):
        sys.exit(1)

    if mode == "quick":
        print("\n[+] Quick Evaluation completed successfully.")
        return

    # 2. Medium Evaluation (DNA Structure awareness + Stop Codon Placement)
    print("\n>>> Phase 2: DNA Shape & stop codon placement <<<")

    # Run structural awareness probe
    cmd_struct = [
        python_bin, "-m", "scripts.probe_structural_awareness",
        run_id
    ]
    if not run_cmd(cmd_struct):
        print("[!] Warning: Structural awareness probe encountered an issue.")

    # Run prefix generation stop-codon placement check
    cmd_gen = [
        python_bin, "-m", "scripts.eval_generation_prefix",
        "--run_id", run_id,
        "--max_genes", str(args.max_genes),
        "--samples", str(args.samples),
        "--max_new", "100"
    ]
    if not run_cmd(cmd_gen, env=env):
        sys.exit(1)

    if mode == "medium":
        print("\n[+] Medium Evaluation completed successfully.")
        return

    # 3. Full Evaluation (SOTA Zero-Shot + Essentiality Linear Probes)
    print("\n>>> Phase 3: Downstream SOTA Benchmarking & Reports <<<")

    # Run zero-shot mutation Spearman correlation
    cmd_mut = [
        python_bin, "-m", "scripts.benchmark_zero_shot_mutations",
        "--run_id", run_id,
        "--device", device
    ]
    if not run_cmd(cmd_mut):
        sys.exit(1)

    # Run gene essentiality linear probes
    cmd_ess = [
        python_bin, "-m", "scripts.benchmark_gene_essentiality",
        "--run_id", run_id,
        "--device", device
    ]
    if not run_cmd(cmd_ess):
        sys.exit(1)
    # Run structural regression probes
    cmd_reg_probe = [
        python_bin, "-m", "scripts.probe_structural_regression",
        "--run_id", run_id
    ]
    # Enforce CPU execution for structural regression probe to prevent Apple Silicon MPS lockups
    if not run_cmd(cmd_reg_probe, env={"FORCE_CPU": "1"}):
        print("[!] Warning: Structural regression probe encountered an issue.")

    # Generate unified SOTA comparison report
    cmd_report = [
        python_bin, "-m", "scripts.generate_sota_report",
        "--run_id", run_id
    ]
    if not run_cmd(cmd_report):
        sys.exit(1)

    print("\n[+] Full Evaluation Suite completed successfully. SOTA report generated.")

if __name__ == "__main__":
    main()
