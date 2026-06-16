#!/bin/bash
# Script to sequentially train 6L4H_d384 (transfer from 4L2H_d384) and 10L8H_d384 (transfer from 6L4H_d384)
# Also runs test set evaluations and biological sanity KPIs on CPU.
# Wrapped under `caffeinate -i` to prevent system sleep during training cycles.

set -e

# Detect today's date
TODAY=$(date +%Y-%m-%d)
echo "=== Starting Transfer Learning Chain for ${TODAY} ==="

RUN_ID_6L4H="${TODAY}_stage2.5_6L4H_d384_e5"
RUN_ID_10L8H="${TODAY}_stage2.5_10L8H_d384_e5"

# ----------------------------------------------------
# 0. Wait for active training run to complete
# ----------------------------------------------------
ACTIVE_PID=60764
if kill -0 $ACTIVE_PID 2>/dev/null; then
  echo ">>> Waiting for active training process (PID: $ACTIVE_PID) to complete..."
  while kill -0 $ACTIVE_PID 2>/dev/null; do
    sleep 60
  done
  echo ">>> Active training process finished. Starting transfer chain."
else
  echo ">>> Active training process (PID: $ACTIVE_PID) not found. Proceeding immediately."
fi

# ----------------------------------------------------
# 1. Train & Evaluate 6L4H Model
# ----------------------------------------------------
echo ">>> [1/2] Training 6L4H Model: ${RUN_ID_6L4H} (Caffeinated)..."
caffeinate -i python -m src.codonlm.train_codon_lm \
  --config configs/stage2.5_6L4H_d384_transfer.yaml \
  --run_id "${RUN_ID_6L4H}"

# Copy vocabulary to the run directory (required for evaluation)
mkdir -p "runs/${RUN_ID_6L4H}"
cp data/processed/itos_codon.txt "runs/${RUN_ID_6L4H}/itos.txt"

echo ">>> [1/2] Evaluating 6L4H Test Perplexity..."
FORCE_CPU=1 python -m scripts.evaluate_test \
  --run_dir "runs/${RUN_ID_6L4H}" \
  --data_dir data/processed/stage2.5_master_pack_v2

echo ">>> [1/2] Computing 6L4H Sanity KPIs..."
FORCE_CPU=1 python -m scripts.sanity_kpis \
  --run_dir "runs/${RUN_ID_6L4H}" \
  --test_npz data/processed/stage2.5_master_pack_v2/test_bs512.npz

# ----------------------------------------------------
# 2. Train & Evaluate 10L8H Model
# ----------------------------------------------------
echo ">>> [2/2] Training 10L8H Model: ${RUN_ID_10L8H} (Caffeinated)..."
# We override --transfer_from to point to the newly trained 6L4H best checkpoint
caffeinate -i python -m src.codonlm.train_codon_lm \
  --config configs/stage2.5_10L8H_d384_transfer.yaml \
  --run_id "${RUN_ID_10L8H}" \
  --transfer_from "runs/${RUN_ID_6L4H}/checkpoints/best.pt"

# Copy vocabulary to the run directory (required for evaluation)
mkdir -p "runs/${RUN_ID_10L8H}"
cp data/processed/itos_codon.txt "runs/${RUN_ID_10L8H}/itos.txt"

echo ">>> [2/2] Evaluating 10L8H Test Perplexity..."
FORCE_CPU=1 python -m scripts.evaluate_test \
  --run_dir "runs/${RUN_ID_10L8H}" \
  --data_dir data/processed/stage2.5_master_pack_v2

echo ">>> [2/2] Computing 10L8H Sanity KPIs..."
FORCE_CPU=1 python -m scripts.sanity_kpis \
  --run_dir "runs/${RUN_ID_10L8H}" \
  --test_npz data/processed/stage2.5_master_pack_v2/test_bs512.npz

echo "=== Transfer Learning Chain Completed Successfully ==="
