#!/usr/bin/env bash
# Post-process only: collect artifacts and organize run outputs under runs/<RUN_ID>.
# Usage: ./post_process.sh RUN_ID [CONFIG]
# Default CONFIG: configs/tiny_mps.yaml

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 RUN_ID [CONFIG]" >&2
  exit 1
fi

RUN_ID="$1"
CONFIG="${2:-configs/tiny_mps.yaml}"

python -m scripts.collect_artifacts_yaml "${RUN_ID}" "${CONFIG}"

# Copy mutation scores into run dir if present to help summarizers
RUN_DIR="runs/$RUN_ID"
SCORES_FILE="outputs/scores/$RUN_ID/one_cds__best.tsv"
if [ -f "$SCORES_FILE" ]; then
  mkdir -p "$RUN_DIR"
  cp "$SCORES_FILE" "$RUN_DIR/one_cds__best.tsv"
fi

echo "[post-process] Collected artifacts for ${RUN_ID} into ${RUN_DIR}"

