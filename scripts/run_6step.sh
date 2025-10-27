#!/usr/bin/env bash
# Run the 6-step interpretability analysis for a given RUN_ID.
# Usage: scripts/run_6step.sh RUN_ID [CONFIG]
# Default CONFIG: configs/tiny_mps_v2.yaml

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 RUN_ID [CONFIG]" >&2
  exit 1
fi

RUN_ID="$1"
CONFIG="${2:-configs/tiny_mps_v2.yaml}"
SAL_WINDOW=$(python -c "import yaml;cfg=yaml.safe_load(open('$CONFIG')) or {};print(int(cfg.get('saliency_window',9)))")
SAL_TOP=$(python -c "import yaml;cfg=yaml.safe_load(open('$CONFIG')) or {};print(int(cfg.get('saliency_top',20)))")

echo "[6step] run_id=${RUN_ID} config=${CONFIG}"

# 0) Collect artifacts
python -m scripts.collect_artifacts_yaml "${RUN_ID}" "${CONFIG}"

# 1) Frequencies
python -m scripts.analyze_frequencies   "${RUN_ID}"

# 2) Embeddings (PCA + neighbors)
python -m scripts.analyze_embeddings    "${RUN_ID}"

# 3) Attention heatmaps
python -m scripts.analyze_attention     "${RUN_ID}"

# 4) Next-token probes
python -m scripts.probe_next_token      "${RUN_ID}"

# 5) Saliency
python -m scripts.analyze_saliency      "${RUN_ID}"
python -m scripts.report_top_saliency   "${RUN_ID}" --window "${SAL_WINDOW}" --top "${SAL_TOP}"

# 6) Biology-aware linear probes
python -m scripts.generate_probe_labels "${RUN_ID}"
python -m scripts.probe_linear          "${RUN_ID}"

# Optional: summarize single-CDS TSV and export a compact LLM-ready JSON
python -m scripts.summarize_one_cds     "${RUN_ID}" || true
python -m scripts.export_run_summary    "${RUN_ID}" || true

echo "[6step] Done. See runs/${RUN_ID}/{charts,tables} and runs/${RUN_ID}/llm_summary.json"
