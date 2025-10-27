#!/usr/bin/env bash
# End-to-end pipeline: data prep + training + quick eval
# Replaces the previous monolithic pipeline.sh implementation.

eval "$(conda shell.bash hook)" || true
conda activate codonlm || true

set -euo pipefail

usage() {
  cat >&2 <<USAGE
Usage: $0 [-c|--config PATH] [-r|--resume CHECKPOINT] [--dataset NAME,GBFF[,MIN_LEN]] [--force]
USAGE
  exit 1
}

format_duration() {
  local total=$1
  local hours=$((total / 3600))
  local minutes=$(((total % 3600) / 60))
  local seconds=$((total % 60))
  printf "%d hours, %d minutes, %d seconds" "$hours" "$minutes" "$seconds"
}

DEFAULT_CONF="configs/tiny_mps.yaml"
CONF="$DEFAULT_CONF"
RESUME=""
FORCE=0
EXTRA_DATASETS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--config)
      [[ $# -lt 2 ]] && { echo "[error] --config requires a path" >&2; usage; }
      CONF="$2"; shift 2 ;;
    -r|--resume)
      [[ $# -lt 2 ]] && { echo "[error] --resume requires a path" >&2; usage; }
      RESUME="$2"; shift 2 ;;
    --dataset)
      [[ $# -lt 2 ]] && { echo "[error] --dataset requires NAME,GBFF[,MIN_LEN]" >&2; usage; }
      EXTRA_DATASETS+=("$2"); shift 2 ;;
    --force)
      FORCE=1; shift ;;
    -h|--help)
      usage ;;
    *) echo "[error] Unknown argument: $1" >&2; usage ;;
  esac
done

if [[ ! -f "$CONF" ]]; then
  echo "[error] Config file not found: $CONF" >&2
  usage
fi

if [[ -n "$RESUME" && ! -f "$RESUME" ]]; then
  echo "[error] Resume checkpoint not found: $RESUME" >&2
  usage
fi

if [[ -z "${RUN_ID:-}" && -n "$RESUME" ]]; then
  if [[ "$RESUME" == */outputs/checkpoints/*/* ]]; then
    RUN_ID=$(basename "$(dirname "$RESUME")")
  fi
fi

# Auto-generate RUN_ID from config if not provided
RUN_ID=${RUN_ID:-$(python -m scripts.make_run_id "$CONF")}
RUN_DIR="runs/${RUN_ID}"
mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/log.txt"

echo "[info] run_id=${RUN_ID}" | tee "$LOG"
echo "[info] config=${CONF}" | tee -a "$LOG"
echo "[info] resume=${RESUME:-none}" | tee -a "$LOG"
echo "[hardware] date: $(date -u +"%Y-%m-%d %H:%M:%S UTC")" | tee -a "$LOG"
echo "[hardware] uname: $(uname -a)" | tee -a "$LOG"
if command -v sysctl >/dev/null 2>&1; then
  sysctl -n machdep.cpu.brand_string 2>/dev/null | sed 's/^/[hardware] cpu: /' | tee -a "$LOG" || true
fi
python - <<'PY' | tee -a "$LOG"
import torch, platform
print(f"[hardware] python: {platform.python_version()}")
print(f"[hardware] torch: {getattr(torch, '__version__', 'NA')}")
print(f"[hardware] mps_available: {torch.backends.mps.is_available()}")
print(f"[hardware] cuda_available: {torch.cuda.is_available()}")
print(f"[hardware] cuda_device_count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print("[hardware] cuda_device: ", torch.cuda.get_device_name(0))
PY

echo "[config] snapshot:" | tee -a "$LOG"
sed 's/^/[config] /' "$CONF" | tee -a "$LOG"
echo "[info] extra_datasets_cli=${EXTRA_DATASETS[*]:-none}" | tee -a "$LOG"

T0=$(date +%s)

# Prepare datasets via Python helper
PREP_ARGS=(--config "$CONF" --run-id "$RUN_ID" --run-dir "$RUN_DIR")
if [[ $FORCE -eq 1 ]]; then PREP_ARGS+=(--force); fi
if [[ ${#EXTRA_DATASETS[@]} -gt 0 ]]; then
  for spec in "${EXTRA_DATASETS[@]}"; do PREP_ARGS+=(--extra-dataset "$spec"); done
fi
python -m scripts.pipeline_prepare "${PREP_ARGS[@]}" 2>&1 | tee -a "$LOG"

PREP_JSON="${RUN_DIR}/pipeline_prepare.json"
if [[ ! -f "$PREP_JSON" ]]; then
  echo "[error] pipeline_prepare did not produce ${PREP_JSON}" | tee -a "$LOG"
  exit 1
fi

eval "$(
python - <<'PY' "$PREP_JSON"
import json, shlex, sys
info = json.load(open(sys.argv[1]))
mapping = {
    "TRAIN_NPZ": info["train_npz"],
    "VAL_NPZ": info["val_npz"],
    "TEST_NPZ": info["test_npz"],
    "PRIMARY_DNA": info["primary_dna"],
    "COMBINED_MANIFEST": info["combined_manifest"],
}
for key, value in mapping.items():
    print(f'{key}={shlex.quote(str(value))}')
PY
)"
echo "[info] combined_manifest=${COMBINED_MANIFEST}" | tee -a "$LOG"
echo "[info] train_npz=${TRAIN_NPZ}" | tee -a "$LOG"
echo "[info] val_npz=${VAL_NPZ}" | tee -a "$LOG"
echo "[info] test_npz=${TEST_NPZ}" | tee -a "$LOG"
echo "[info] primary_dna=${PRIMARY_DNA}" | tee -a "$LOG"

# Train
CKPT_ROOT="outputs/checkpoints/${RUN_ID}"
SCORES_ROOT="outputs/scores/${RUN_ID}"
Ttrain0=$(date +%s)
TRAIN_ARGS=(--config "$CONF" --run_id "${RUN_ID}" --train_npz "$TRAIN_NPZ" --val_npz "$VAL_NPZ" --test_npz "$TEST_NPZ")
if [[ -n "$RESUME" ]]; then TRAIN_ARGS+=(--resume "$RESUME"); fi
python -m src.codonlm.train_codon_lm "${TRAIN_ARGS[@]}" 2>&1 | tee -a "$LOG"
Ttrain1=$(date +%s)

# Evaluate on val and test sets
python -m src.codonlm.eval_perplexity --ckpt "${CKPT_ROOT}/best.pt" --val_npz "$VAL_NPZ" 2>&1 | tee -a "$LOG" || true
python -m src.codonlm.eval_perplexity --ckpt "${CKPT_ROOT}/best.pt" --val_npz "$TEST_NPZ" 2>&1 | tee -a "$LOG" || true

# Score mutations for one CDS
head -n1 "$PRIMARY_DNA" > data/processed/one_cds.txt
mkdir -p "${SCORES_ROOT}"
conda run -n codonlm python -m src.codonlm.score_mutations --ckpt "${CKPT_ROOT}/best.pt" --dna data/processed/one_cds.txt --out "${SCORES_ROOT}/one_cds__best.tsv" 2>&1 | tee -a "$LOG" || true

# Mine motifs (quick)
python -m src.codonlm.mine_motifs --ckpt "${CKPT_ROOT}/best.pt" --npz "$TRAIN_NPZ" --k 9 --clusters 100 2>&1 | tee -a "$LOG" || true
if [ -f outputs/motif_clusters.npz ]; then
  cp outputs/motif_clusters.npz "$RUN_DIR/motif_clusters.npz" || true
fi

# Collect artifacts under runs/<RUN_ID>
python -m scripts.collect_artifacts_yaml "${RUN_ID}" "$CONF" 2>&1 | tee -a "$LOG" || true

T1=$(date +%s)
TRAIN_SEC=$((Ttrain1-Ttrain0))
TOTAL_SEC=$((T1-T0))
echo "[timing] training_sec=${TRAIN_SEC}" | tee -a "$LOG"
echo "[timing] training_time=$(format_duration "$TRAIN_SEC")" | tee -a "$LOG"
echo "[timing] total_sec=${TOTAL_SEC}" | tee -a "$LOG"
echo "[timing] total_time=$(format_duration "$TOTAL_SEC")" | tee -a "$LOG"
