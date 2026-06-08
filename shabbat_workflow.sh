#!/usr/bin/env bash
# Shabbat Workflow: 26-Hour Automated Pipeline

set -euo pipefail

LOG_FILE="outputs/shabbat_workflow.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=================================================="
echo "Starting Shabbat Automated Workflow"
echo "Date: $(date)"
echo "=================================================="

# 1. Wait for current training (PID: 90542)
echo "[*] Waiting for current Master training (PID: 90542) to complete..."
while kill -0 90542 2> /dev/null; do
    sleep 300
done
echo "[*] Master training complete."

# 2. Evaluate
echo "[*] Evaluating Master Run..."
RUN_ID="2026-06-05_stage2.5_6L4H_d256_e10"
PYTHONPATH=. python -m scripts.eval_generation_prefix --run_id $RUN_ID --k_list 1,3,5 --samples 10 --max_genes 20 --max_new 100 --target_aa_len 100 --require_terminal_stop || true
PYTHONPATH=. python scripts/benchmark_motifs.py $RUN_ID || true
PYTHONPATH=. python scripts/audit_structural_motifs.py $RUN_ID || true
PYTHONPATH=. python scripts/generate_plain_english_report.py $RUN_ID || true

# 3. Extended Patience Run
echo "[*] Launching Extended Master Run (10 more epochs)..."
PYTHONPATH=. python -m src.codonlm.train_codon_lm --config configs/stage2.5_extended.yaml || true
echo "[*] Extended Run Complete."

# 4. M2-Max Contingency
echo "[*] Launching M2-Max 10-Layer Contingency..."
PYTHONPATH=. python -m src.codonlm.train_codon_lm --config configs/m2_max_10L8H.yaml || true
echo "[*] M2-Max Run Complete."

echo "=================================================="
echo "Shabbat Workflow Complete"
echo "Date: $(date)"
echo "=================================================="
