#!/usr/bin/env bash
# Run the 6-step interpretability analysis for a given RUN_ID.
# Usage: ./analysis.sh RUN_ID [CONFIG]
# Default CONFIG: configs/tiny_mps.yaml

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 RUN_ID [CONFIG]" >&2
  exit 1
fi

RUN_ID="$1"
CONFIG="${2:-configs/tiny_mps.yaml}"
SAL_WINDOW=$(python -c "import yaml;cfg=yaml.safe_load(open('$CONFIG')) or {};print(int(cfg.get('saliency_window',9)))")
SAL_TOP=$(python -c "import yaml;cfg=yaml.safe_load(open('$CONFIG')) or {};print(int(cfg.get('saliency_top',20)))")

echo "[analysis] run_id=${RUN_ID} config=${CONFIG}"

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

# Additional evaluation and quality checks
# 7) Test perplexity + KPIs
python -m scripts.evaluate_test --run_dir "outputs/checkpoints/${RUN_ID}" || true
python -m scripts.sanity_kpis --run_dir "outputs/checkpoints/${RUN_ID}" || true

# 8) Sequence quality, SS/disorder, calibration
VAL_NPZ=$(python - <<'PY'
import json,sys,Pathlib
from pathlib import Path
run_id = sys.argv[1]
prep = Path('runs')/run_id/'pipeline_prepare.json'
try:
    info = json.loads(prep.read_text())
    print(info.get('val_npz',''))
except Exception:
    print('')
PY
"${RUN_ID}")
python -m scripts.seq_quality --run_id "${RUN_ID}" || true
python -m scripts.ss_propensity --run_id "${RUN_ID}" || true
python -m scripts.disorder_heuristics --run_id "${RUN_ID}" || true
if [[ -n "${VAL_NPZ}" && -f "${VAL_NPZ}" ]]; then
  python -m scripts.calibration_metrics --ckpt "outputs/checkpoints/${RUN_ID}/best.pt" --npz "${VAL_NPZ}" --out "outputs/scores/${RUN_ID}/metrics.json" || true
fi

# 9) Optional: prefix generation summary plots if present
SUMMARY="outputs/scores/${RUN_ID}/gen_prefix/summary.csv"
if [[ -f "${SUMMARY}" ]]; then
  python -m scripts.plot_eval_prefix --summary "${SUMMARY}" --out_dir "outputs/figs" || true
fi

echo "[analysis] Done. See runs/${RUN_ID}/{charts,tables} and runs/${RUN_ID}/llm_summary.json"
