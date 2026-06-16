#!/usr/bin/env python3
"""
Compares best.pt and last.pt checkpoints of a run on the full evaluation suite:
1. Validation/Test Perplexity
2. DNAshape Structural Regression
3. DMS Zero-Shot Mutation Fitness (Spearman Rho)
4. Gene Essentiality Classification (Accuracy, F1, MCC)

Generates a markdown comparison report.
"""

import argparse
import subprocess
import shutil
import json
import sys
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
        return False

def get_metrics(metrics_path):
    if metrics_path.exists():
        try:
            with open(metrics_path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--device", default="mps", help="Device to run evaluations on (mps, cpu, cuda)")
    args = ap.parse_args()

    run_id = args.run_id
    device = args.device
    run_dir = Path("runs") / run_id
    if not run_dir.exists():
        print(f"[!] Run directory {run_dir} does not exist.")
        sys.exit(1)

    checkpoint_dir = run_dir / "checkpoints"
    scores_dir = run_dir / "scores"
    metrics_path = scores_dir / "metrics.json"

    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"

    if not best_path.exists():
        print(f"[!] best.pt not found in {checkpoint_dir}")
        sys.exit(1)
    if not last_path.exists():
        print(f"[!] last.pt not found in {checkpoint_dir}")
        sys.exit(1)

    python_bin = sys.executable

    # Back up original best.pt and metrics.json
    print("[*] Creating backups for best.pt and metrics.json...")
    backup_best = checkpoint_dir / "best.pt.backup"
    backup_metrics = scores_dir / "metrics.json.backup"

    shutil.copy2(best_path, backup_best)
    original_metrics_exist = metrics_path.exists()
    if original_metrics_exist:
        shutil.copy2(metrics_path, backup_metrics)

    best_metrics = {}
    last_metrics = {}

    # Set up device environments
    eval_env = {}
    if device == "cpu":
        eval_env["FORCE_CPU"] = "1"

    try:
        # 1. Run full evaluation suite on best.pt
        print(f"\n[*] Evaluating best.pt on device {device}...")
        if metrics_path.exists():
            metrics_path.unlink()

        run_cmd([python_bin, "-m", "scripts.evaluate_test", "--run_id", run_id], env=eval_env)
        run_cmd([python_bin, "-m", "scripts.probe_structural_regression", "--run_id", run_id, "--no_cache"], env=eval_env)
        run_cmd([python_bin, "-m", "scripts.benchmark_zero_shot_mutations", "--run_id", run_id, "--device", device])
        run_cmd([python_bin, "-m", "scripts.benchmark_gene_essentiality", "--run_id", run_id, "--device", device])

        best_metrics = get_metrics(metrics_path).copy()

        # 2. Run full evaluation suite on last.pt using temporary file swapping
        print(f"\n[*] Evaluating last.pt on device {device}...")
        shutil.copy2(last_path, best_path)

        if metrics_path.exists():
            metrics_path.unlink()

        run_cmd([python_bin, "-m", "scripts.evaluate_test", "--run_id", run_id], env=eval_env)
        run_cmd([python_bin, "-m", "scripts.probe_structural_regression", "--run_id", run_id, "--no_cache"], env=eval_env)
        run_cmd([python_bin, "-m", "scripts.benchmark_zero_shot_mutations", "--run_id", run_id, "--device", device])
        run_cmd([python_bin, "-m", "scripts.benchmark_gene_essentiality", "--run_id", run_id, "--device", device])

        last_metrics = get_metrics(metrics_path).copy()

    finally:
        # Restore original best.pt and metrics.json
        print("[*] Restoring best.pt and original metrics...")
        if backup_best.exists():
            shutil.move(backup_best, best_path)
        if original_metrics_exist and backup_metrics.exists():
            shutil.move(backup_metrics, metrics_path)

    # 3. Generate Markdown Comparison Report
    report_path = scores_dir / "best_vs_last_comparison.md"

    # Extract specific values
    def val(metrics_dict, key):
        if key == "test_perplexity" and "test_ppl" in metrics_dict:
            key = "test_ppl"
        return f"{metrics_dict.get(key, 0.0):.4f}" if key in metrics_dict else "N/A"

    def get_f(metrics_dict, key):
        if key == "test_perplexity" and "test_ppl" in metrics_dict:
            key = "test_ppl"
        return float(metrics_dict.get(key, 0.0))

    report_content = f"""# Checkpoint Comparison Report: best.pt vs. last.pt
**Run ID:** `{run_id}`

This report compares the best checkpoint (based on lowest validation loss) against the last training checkpoint to see if additional training epochs improved downstream biological representation, even if loss/perplexity slightly degraded.

---

## 📊 Performance Metrics Comparison

| Benchmark Metric | best.pt | last.pt | Delta |
| :--- | :---: | :---: | :---: |
| **Test Loss (NLL)** | {val(best_metrics, "test_loss")} | {val(last_metrics, "test_loss")} | {get_f(last_metrics, "test_loss") - get_f(best_metrics, "test_loss"):+.4f} |
| **Test Perplexity** | {val(best_metrics, "test_perplexity")} | {val(last_metrics, "test_perplexity")} | {get_f(last_metrics, "test_perplexity") - get_f(best_metrics, "test_perplexity"):+.4f} |
| **DNAshape Avg $R^2$** | {val(best_metrics, "regression_probe.avg_r2")} | {val(last_metrics, "regression_probe.avg_r2")} | {get_f(last_metrics, "regression_probe.avg_r2") - get_f(best_metrics, "regression_probe.avg_r2"):+.4f} |
| **DNAshape Avg Corr ($\rho$)** | {val(best_metrics, "regression_probe.avg_corr")} | {val(last_metrics, "regression_probe.avg_corr")} | {get_f(last_metrics, "regression_probe.avg_corr") - get_f(best_metrics, "regression_probe.avg_corr"):+.4f} |
| **Lambda Essentiality ACC** | {val(best_metrics, "sota_lambda_essentiality_acc")} | {val(last_metrics, "sota_lambda_essentiality_acc")} | {get_f(last_metrics, "sota_lambda_essentiality_acc") - get_f(best_metrics, "sota_lambda_essentiality_acc"):+.4f} |
| **Lambda Essentiality F1** | {val(best_metrics, "sota_lambda_essentiality_f1")} | {val(last_metrics, "sota_lambda_essentiality_f1")} | {get_f(last_metrics, "sota_lambda_essentiality_f1") - get_f(best_metrics, "sota_lambda_essentiality_f1"):+.4f} |
| **Lambda Essentiality MCC** | {val(best_metrics, "sota_lambda_essentiality_mcc")} | {val(last_metrics, "sota_lambda_essentiality_mcc")} | {get_f(last_metrics, "sota_lambda_essentiality_mcc") - get_f(best_metrics, "sota_lambda_essentiality_mcc"):+.4f} |
| **Pseudomonas Ess. ACC** | {val(best_metrics, "sota_pseudomonas_essentiality_acc")} | {val(last_metrics, "sota_pseudomonas_essentiality_acc")} | {get_f(last_metrics, "sota_pseudomonas_essentiality_acc") - get_f(best_metrics, "sota_pseudomonas_essentiality_acc"):+.4f} |
| **Pseudomonas Ess. F1** | {val(best_metrics, "sota_pseudomonas_essentiality_f1")} | {val(last_metrics, "sota_pseudomonas_essentiality_f1")} | {get_f(last_metrics, "sota_pseudomonas_essentiality_f1") - get_f(best_metrics, "sota_pseudomonas_essentiality_f1"):+.4f} |
| **Pseudomonas Ess. MCC** | {val(best_metrics, "sota_pseudomonas_essentiality_mcc")} | {val(last_metrics, "sota_pseudomonas_essentiality_mcc")} | {get_f(last_metrics, "sota_pseudomonas_essentiality_mcc") - get_f(best_metrics, "sota_pseudomonas_essentiality_mcc"):+.4f} |
| **Protein DMS Spearman** | {val(best_metrics, "sota_protein_dms_spearman")} | {val(last_metrics, "sota_protein_dms_spearman")} | {get_f(last_metrics, "sota_protein_dms_spearman") - get_f(best_metrics, "sota_protein_dms_spearman"):+.4f} |
| **rRNA DMS Spearman** | {val(best_metrics, "sota_rrna_dms_spearman")} | {val(last_metrics, "sota_rrna_dms_spearman")} | {get_f(last_metrics, "sota_rrna_dms_spearman") - get_f(best_metrics, "sota_rrna_dms_spearman"):+.4f} |

---

## 💡 Key Insights
* **Loss vs. Biology**: Check if the Spearman rank correlation coefficients or downstream essentiality MCC values improved on `last.pt` despite a potential increase in test loss/perplexity.
* **Biophysical Latent Manifold**: Note if the DNAshape regression average $R^2$ score improved or stayed stable, confirming the structural integrity of representations over training time.
"""

    report_path.write_text(report_content)
    print(f"\n[+] Saved comparison report to: {report_path}")

if __name__ == "__main__":
    main()
