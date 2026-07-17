#!/usr/bin/env python3
"""
Runs a structured ablation sweep across 8 configurations evaluating:
1. Baseline vs Biophysical Shape-Guided models.
2. Unguided vs EBM-guided decoding.
3. No stop bias vs ReD stop-bias decoding.
"""

import subprocess
import pandas as pd
from pathlib import Path

# Paths
REPO_DIR = Path(__file__).resolve().parents[1]
RUN_BASELINE = "2026-06-19_physical_termination_transfer_mps_b4_e1"
RUN_BIOPHYS = "2026-07-13_physical_termination_shape_guided_e1_v2"

CONFIGS = [
    # 1. Baseline Unguided
    {
        "name": "Baseline Unguided",
        "run_id": RUN_BASELINE,
        "flags": []
    },
    # 2. Baseline + ReD
    {
        "name": "Baseline + ReD Bias",
        "run_id": RUN_BASELINE,
        "flags": ["--termination_bias", "--termination_stop_bias", "8.0", "--termination_bias_window", "50"]
    },
    # 3. Baseline + EBM
    {
        "name": "Baseline + EBM",
        "run_id": RUN_BASELINE,
        "flags": ["--ebm_guidance", "--guide_alpha", "1.0"]
    },
    # 4. Baseline + EBM + ReD
    {
        "name": "Baseline + EBM + ReD",
        "run_id": RUN_BASELINE,
        "flags": ["--ebm_guidance", "--guide_alpha", "1.0", "--termination_bias", "--termination_stop_bias", "8.0", "--termination_bias_window", "50"]
    },
    # 5. Biophysical Unguided
    {
        "name": "Biophysical Unguided",
        "run_id": RUN_BIOPHYS,
        "flags": []
    },
    # 6. Biophysical + ReD
    {
        "name": "Biophysical + ReD Bias",
        "run_id": RUN_BIOPHYS,
        "flags": ["--termination_bias", "--termination_stop_bias", "8.0", "--termination_bias_window", "50"]
    },
    # 7. Biophysical + EBM
    {
        "name": "Biophysical + EBM",
        "run_id": RUN_BIOPHYS,
        "flags": ["--ebm_guidance", "--guide_alpha", "1.0"]
    },
    # 8. Biophysical + EBM + ReD
    {
        "name": "Biophysical + EBM + ReD",
        "run_id": RUN_BIOPHYS,
        "flags": ["--ebm_guidance", "--guide_alpha", "1.0", "--termination_bias", "--termination_stop_bias", "8.0", "--termination_bias_window", "50"]
    }
]

def run_configuration(cfg_idx: int, config: dict):
    name = config["name"]
    label = name.lower().replace(" ", "_").replace("+", "plus")
    run_id = config["run_id"]
    
    cmd = [
        "python", "-m", "scripts.eval_generation_prefix",
        "--run_id", run_id,
        "--preset", "quick",
        "--seed", "1337",
        "--allow_non_cds_tokens",
        "--out_label", label
    ] + config["flags"]
    
    print(f"\n==================================================")
    print(f"[{cfg_idx + 1}/8] Running: {name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"==================================================")
    
    subprocess.run(cmd, check=True)

def collect_results():
    results = []
    for config in CONFIGS:
        name = config["name"]
        label = name.lower().replace(" ", "_").replace("+", "plus")
        run_id = config["run_id"]
        
        summary_path = REPO_DIR / "runs" / run_id / "scores" / label / "summary.csv"
        if summary_path.exists():
            df = pd.read_csv(summary_path)
            # Take the metrics for k=3 (line where k==3)
            row_k3 = df[df["k"] == 3].iloc[0]
            results.append({
                "Configuration": name,
                "Mean AA Len": f"{row_k3['mean_aa_len']:.2f}",
                "Raw Mean AA Len": f"{row_k3.get('raw_mean_aa_len', row_k3['mean_aa_len']):.2f}",
                "Median GQS": f"{row_k3['median_gqs']:.2f}",
                "Raw Median GQS": f"{row_k3.get('raw_median_gqs', row_k3['median_gqs']):.2f}",
                "Hard Cap Rate": f"{row_k3['hard_cap_rate']:.2f}",
                "Raw Hard Cap Rate": f"{row_k3.get('raw_hard_cap_rate', row_k3['hard_cap_rate']):.2f}",
                "Termination Rate": f"{row_k3['termination_rate']:.2f}",
                "Raw Termination Rate": f"{row_k3.get('raw_termination_rate', row_k3['termination_rate']):.2f}"
            })
        else:
            results.append({
                "Configuration": name,
                "Mean AA Len": "N/A",
                "Raw Mean AA Len": "N/A",
                "Median GQS": "N/A",
                "Raw Median GQS": "N/A",
                "Hard Cap Rate": "N/A",
                "Raw Hard Cap Rate": "N/A",
                "Termination Rate": "N/A",
                "Raw Termination Rate": "N/A"
            })
            
    # Print as Markdown Table
    df_res = pd.DataFrame(results)
    md_table = df_res.to_markdown(index=False)
    print("\n\n### Ablation Sweep Results (k=3)")
    print(md_table)
    
    # Save to artifacts
    report_path = REPO_DIR / "runs" / "ablation_report.md"
    report_path.write_text(f"# Ablation Matrix Results\n\nGenerated automatically via `run_ablation_sweep.py`.\n\n{md_table}\n")
    print(f"\nWrote report to: {report_path}")

def main():
    for idx, config in enumerate(CONFIGS):
        run_configuration(idx, config)
    collect_results()

if __name__ == "__main__":
    main()
