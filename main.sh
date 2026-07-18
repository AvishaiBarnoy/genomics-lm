#!/usr/bin/env bash
# End-to-end pipeline: data prep & training suite.
# Note: This script manages dataset pre-processing and model training workflows.
# It does NOT handle downstream sequence generation/inference. For sequence
# generation, use scripts/generative_design_loop.py.

activate_codonlm_conda() {
  if [[ "${CODONLM_SKIP_CONDA:-0}" == "1" ]]; then
    return
  fi
  if ! command -v conda >/dev/null 2>&1; then
    return
  fi
  if ! conda env list 2>/dev/null | awk '$1 == "codonlm" { found=1 } END { exit !found }'; then
    echo "[info] conda environment 'codonlm' not found; using current Python environment" >&2
    return
  fi
  eval "$(conda shell.bash hook)"
  conda activate codonlm
}

activate_codonlm_conda

set -euo pipefail

usage() {
  cat >&2 <<USAGE
Usage: $0 [-c|--config PATH] [-r|--resume CHECKPOINT] [--dataset NAME,GBFF[,MIN_LEN]] [--force] [--allow-sequence-split] [--with-artifacts] [--with-motifs] [--preprocess-only] [--dry-run]
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
ALLOW_SEQUENCE_SPLIT=0
RUN_ID_WAS_SET=0
if [[ -n "${RUN_ID:-}" ]]; then
  RUN_ID_WAS_SET=1
fi
EXTRA_DATASETS=()
WITH_ARTIFACTS=0
WITH_MOTIFS=0
PREPROCESS_ONLY=0
DRY_RUN=0

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
    --allow-sequence-split)
      ALLOW_SEQUENCE_SPLIT=1; shift ;;
    --with-artifacts)
      WITH_ARTIFACTS=1; shift ;;
    --with-motifs)
      WITH_MOTIFS=1; shift ;;
    --preprocess-only)
      PREPROCESS_ONLY=1; shift ;;
    --dry-run)
      DRY_RUN=1; shift ;;
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

# Resolve trainer type from config (defaulting to codon_lm)
TRAINER=$(python -c "
import yaml
try:
    with open('$CONF') as f:
        cfg = yaml.safe_load(f) or {}
    task = cfg.get('task') or {}
    trainer = cfg.get('trainer') or (task.get('trainer') if isinstance(task, dict) else None) or 'codon_lm'
    print(str(trainer).strip().lower())
except Exception:
    print('codon_lm')
")

if [[ "$TRAINER" != "codon_lm" && "$TRAINER" != "protein_lm" && "$TRAINER" != "protein_multitask" && "$TRAINER" != "protein_classifier" ]]; then
  echo "[error] Unknown trainer type: $TRAINER" >&2
  exit 1
fi

if [[ -z "${RUN_ID:-}" && -n "$RESUME" ]]; then
  if [[ "$RESUME" == runs/*/checkpoints/* || "$RESUME" == */runs/*/checkpoints/* ]]; then
    RUN_ID=$(basename "$(dirname "$(dirname "$RESUME")")")
    RUN_ID_WAS_SET=1
  elif [[ "$RESUME" == */outputs/checkpoints/*/* ]]; then
    RUN_ID=$(basename "$(dirname "$RESUME")")
    RUN_ID_WAS_SET=1
  fi
fi

# Auto-generate RUN_ID from config if not provided
RUN_ID=${RUN_ID:-$(python -m scripts.make_run_id "$CONF")}
# Disambiguate RUN_ID if it already exists to avoid overwriting prior runs
BASE_RUN_ID="$RUN_ID"
DISAMBIG=0
if [[ $RUN_ID_WAS_SET -eq 0 && -z "$RESUME" ]]; then
  while [[ -d "runs/${RUN_ID}" || -d "outputs/checkpoints/${RUN_ID}" || -d "outputs/scores/${RUN_ID}" ]]; do
    DISAMBIG=$((DISAMBIG+1))
    RUN_ID="${BASE_RUN_ID}-${DISAMBIG}"
  done
fi
RUN_DIR="runs/${RUN_ID}"
mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/log.txt"

if [[ $DISAMBIG -gt 0 ]]; then
  if [[ -n "$RESUME" ]]; then
    echo "[info] run_id_base=${BASE_RUN_ID}" | tee -a "$LOG"
  else
    echo "[info] run_id_base=${BASE_RUN_ID}" | tee "$LOG"
  fi
  echo "[info] run_id=${RUN_ID} (disambiguated)" | tee -a "$LOG"
else
  if [[ -n "$RESUME" ]]; then
    echo "[info] run_id=${RUN_ID}" | tee -a "$LOG"
  else
    echo "[info] run_id=${RUN_ID}" | tee "$LOG"
  fi
fi
echo "[info] config=${CONF}" | tee -a "$LOG"
echo "[info] trainer=${TRAINER}" | tee -a "$LOG"
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

# Config fingerprint (sha256) and git commit
python - "$CONF" <<'PY' | tee -a "$LOG"
import hashlib, sys, pathlib, subprocess
conf_path = pathlib.Path(sys.argv[1])
try:
    h = hashlib.sha256(conf_path.read_bytes()).hexdigest()
    print(f"[cfg] sha256={h} file={conf_path}")
except Exception as exc:
    print(f"[cfg] sha256=<error> file={conf_path} err={exc}")
try:
    commit = subprocess.check_output(["git","rev-parse","--short","HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    dirty = subprocess.call(["git","diff","--quiet"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"[git] commit={commit} dirty={(dirty!=0)}")
except Exception:
    print("[git] commit=<unknown> dirty=?")
PY

T0=$(date +%s)

# Preprocessing phase
if [[ "$TRAINER" == "codon_lm" ]]; then
  # Check if train_npz dataset paths are already specified in the configuration
  HAS_NPZ=$(python -c "
import yaml
cfg = yaml.safe_load(open('$CONF')) or {}
data = cfg.get('data', {}) if isinstance(cfg.get('data'), dict) else {}
print(1 if ('train_npz' in cfg or 'train_npz' in data) else 0)
")

  if [[ $HAS_NPZ -eq 1 ]]; then
    echo "[info] train_npz found in config. Bypassing data preparation stage." | tee -a "$LOG"
    TRAIN_NPZ=$(python -c "
import yaml
cfg = yaml.safe_load(open('$CONF')) or {}
data = cfg.get('data', {}) if isinstance(cfg.get('data'), dict) else {}
print(cfg.get('train_npz') or data.get('train_npz', ''))
")
    VAL_NPZ=$(python -c "
import yaml
cfg = yaml.safe_load(open('$CONF')) or {}
data = cfg.get('data', {}) if isinstance(cfg.get('data'), dict) else {}
print(cfg.get('val_npz') or data.get('val_npz', ''))
")
    TEST_NPZ=$(python -c "
import yaml
cfg = yaml.safe_load(open('$CONF')) or {}
data = cfg.get('data', {}) if isinstance(cfg.get('data'), dict) else {}
print(cfg.get('test_npz') or data.get('test_npz', ''))
")
    PRIMARY_DNA=""
    COMBINED_MANIFEST=""
  else
    # Prepare all CodonLM records globally before assigning grouped splits.
    PREP_ARGS=(--config "$CONF" --run-id "$RUN_ID" --run-dir "$RUN_DIR")
    if [[ $FORCE -eq 1 ]]; then PREP_ARGS+=(--force); fi
    if [[ $ALLOW_SEQUENCE_SPLIT -eq 1 ]]; then PREP_ARGS+=(--allow-sequence-split); fi
    if [[ ${#EXTRA_DATASETS[@]} -gt 0 ]]; then
      for spec in "${EXTRA_DATASETS[@]}"; do PREP_ARGS+=(--extra-dataset "$spec"); done
    fi
    if [[ $DRY_RUN -eq 1 ]]; then
      echo "[dry-run] Planned dataset preparation: python -m scripts.build_global_manifest ${PREP_ARGS[*]}"
      TRAIN_NPZ="data/processed/global/${RUN_ID}/train_bs512.npz"
      VAL_NPZ="data/processed/global/${RUN_ID}/val_bs512.npz"
      TEST_NPZ="data/processed/global/${RUN_ID}/test_bs512.npz"
      PRIMARY_DNA="data/processed/global/${RUN_ID}/cds_dna.txt"
      COMBINED_MANIFEST="data/processed/global/${RUN_ID}/manifest.json"
    else
      python -m scripts.build_global_manifest "${PREP_ARGS[@]}" 2>&1 | tee -a "$LOG"

      PREP_JSON="${RUN_DIR}/pipeline_prepare.json"
      if [[ ! -f "$PREP_JSON" ]]; then
        echo "[error] build_global_manifest did not produce ${PREP_JSON}" | tee -a "$LOG"
        exit 1
      fi

      eval "$(
      python - "$PREP_JSON" <<'PY'
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
    fi
  fi

else
  # Protein workflows path check
  echo "[info] resolved trainer: $TRAINER. Checking pre-processed datasets existence..." | tee -a "$LOG"
  DATA_PATHS_JSON=$(python -c "
import yaml, json
cfg = yaml.safe_load(open('$CONF')) or {}
data = cfg.get('data', {}) if isinstance(cfg.get('data'), dict) else {}
train_path = cfg.get('train_data') or data.get('train_path') or cfg.get('train_path')
val_path = cfg.get('val_data') or data.get('val_path') or cfg.get('val_path')
print(json.dumps({'train_path': train_path, 'val_path': val_path}))
")
  TRAIN_PATH=$(python -c "import json; print(json.loads('$DATA_PATHS_JSON').get('train_path') or '')")
  VAL_PATH=$(python -c "import json; print(json.loads('$DATA_PATHS_JSON').get('val_path') or '')")

  echo "[info] train_data_path=${TRAIN_PATH:-none}" | tee -a "$LOG"
  echo "[info] val_data_path=${VAL_PATH:-none}" | tee -a "$LOG"

  if [[ -z "$TRAIN_PATH" || -z "$VAL_PATH" ]]; then
    echo "[error] Configuration must specify train and validation paths (e.g. train_data / val_data)." >&2
    exit 1
  fi

  if [[ $DRY_RUN -eq 0 ]]; then
    if [[ ! -f "$TRAIN_PATH" ]]; then
      echo "[error] Preprocessed train dataset not found: $TRAIN_PATH" >&2
      echo "[hint] Please run data extraction first (e.g. python -m scripts.prepare_protein_type_dataset)." >&2
      exit 1
    fi
    if [[ ! -f "$VAL_PATH" ]]; then
      echo "[error] Preprocessed validation dataset not found: $VAL_PATH" >&2
      exit 1
    fi
  fi
fi

# Preprocess-only exit
if [[ $PREPROCESS_ONLY -eq 1 ]]; then
  echo "[info] --preprocess-only requested. Exiting successfully." | tee -a "$LOG"
  if [[ "$TRAINER" == "codon_lm" ]]; then
    echo "Compiled datasets:"
    echo "  train: $TRAIN_NPZ"
    echo "  val:   $VAL_NPZ"
    echo "  test:  $TEST_NPZ"
  else
    echo "Verified protein datasets exist:"
    echo "  train: $TRAIN_PATH"
    echo "  val:   $VAL_PATH"
  fi
  exit 0
fi

if [[ "$TRAINER" == "codon_lm" ]]; then
  echo "[info] combined_manifest=${COMBINED_MANIFEST}" | tee -a "$LOG"
  echo "[info] train_npz=${TRAIN_NPZ}" | tee -a "$LOG"
  echo "[info] val_npz=${VAL_NPZ}" | tee -a "$LOG"
  echo "[info] test_npz=${TEST_NPZ}" | tee -a "$LOG"
  echo "[info] primary_dna=${PRIMARY_DNA}" | tee -a "$LOG"

  # Fingerprint combined manifest contents
  if [[ $DRY_RUN -eq 0 ]]; then
    python - "$COMBINED_MANIFEST" <<'PY' | tee -a "$LOG"
import hashlib, sys, pathlib
p = pathlib.Path(sys.argv[1])
try:
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    print(f"[cfg] combined_manifest.sha256={h}")
except Exception as exc:
    print(f"[cfg] combined_manifest.sha256=<error> err={exc}")
PY
  fi
fi

# Train
CKPT_ROOT="runs/${RUN_ID}/checkpoints"
SCORES_ROOT="runs/${RUN_ID}/scores"
Ttrain0=$(date +%s)

# Training phase dispatch
if [[ "$TRAINER" == "codon_lm" ]]; then
  HAS_BATCH_OPTIMIZER=$(python -c "
import yaml
cfg = yaml.safe_load(open('$CONF')) or {}
section = cfg.get('batch_optimizer') or {}
print(1 if section and section.get('enabled', True) else 0)
")

  TRAIN_ARGS=(--config "$CONF" --run_id "${RUN_ID}" --train_npz "$TRAIN_NPZ" --val_npz "$VAL_NPZ" --test_npz "$TEST_NPZ")
  if [[ -n "$RESUME" ]]; then TRAIN_ARGS+=(--resume "$RESUME"); fi

  if [[ $HAS_BATCH_OPTIMIZER -eq 1 ]]; then
    echo "[train] batch_optimizer enabled; dispatching through scripts.optimize_train_batching" | tee -a "$LOG"
    OPT_ARGS=("${TRAIN_ARGS[@]}" --optimize)
    if [[ $FORCE -eq 1 ]]; then OPT_ARGS+=(--force); fi
    if [[ $DRY_RUN -eq 1 ]]; then
      echo "[dry-run] python -m scripts.optimize_train_batching ${OPT_ARGS[*]}"
    else
      python -m scripts.optimize_train_batching "${OPT_ARGS[@]}" 2>&1 | tee -a "$LOG"
    fi
  else
    echo "[train] batch_optimizer disabled; dispatching directly to src.codonlm.train_codon_lm" | tee -a "$LOG"
    if [[ $DRY_RUN -eq 1 ]]; then
      echo "[dry-run] python -m src.codonlm.train_codon_lm ${TRAIN_ARGS[*]}"
    else
      python -m src.codonlm.train_codon_lm "${TRAIN_ARGS[@]}" 2>&1 | tee -a "$LOG"
    fi
  fi

elif [[ "$TRAINER" == "protein_lm" ]]; then
  echo "[train] resolved trainer protein_lm; dispatching to src.protein_lm.train_lm" | tee -a "$LOG"
  TRAIN_ARGS=(--config "$CONF")
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "[dry-run] python -m src.protein_lm.train_lm ${TRAIN_ARGS[*]}"
  else
    python -m src.protein_lm.train_lm "${TRAIN_ARGS[@]}" 2>&1 | tee -a "$LOG"
  fi

elif [[ "$TRAINER" == "protein_multitask" ]]; then
  echo "[train] resolved trainer protein_multitask; dispatching to src.protein_lm.train_multi_task" | tee -a "$LOG"
  TRAIN_ARGS=(--config "$CONF" --run_id "${RUN_ID}")
  if [[ -n "$RESUME" ]]; then TRAIN_ARGS+=(--resume "$RESUME"); fi
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "[dry-run] python -m src.protein_lm.train_multi_task ${TRAIN_ARGS[*]}"
  else
    python -m src.protein_lm.train_multi_task "${TRAIN_ARGS[@]}" 2>&1 | tee -a "$LOG"
  fi

elif [[ "$TRAINER" == "protein_classifier" ]]; then
  echo "[train] resolved trainer protein_classifier; dispatching to src.protein_lm.train_classifier" | tee -a "$LOG"
  TRAIN_ARGS=(--config "$CONF")
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "[dry-run] python -m src.protein_lm.train_classifier ${TRAIN_ARGS[*]}"
  else
    python -m src.protein_lm.train_classifier "${TRAIN_ARGS[@]}" 2>&1 | tee -a "$LOG"
  fi
fi

Ttrain1=$(date +%s)

# Post-training evaluations (CodonLM only)
if [[ "$TRAINER" == "codon_lm" && $DRY_RUN -eq 0 ]]; then
  # Determine checkpoint to evaluate (fallback to last.pt if best.pt is missing)
  EVAL_CKPT="${CKPT_ROOT}/best.pt"
  if [[ ! -f "$EVAL_CKPT" && -f "${CKPT_ROOT}/last.pt" ]]; then
    echo "[info] best.pt not found; falling back to last.pt for evaluations" | tee -a "$LOG"
    EVAL_CKPT="${CKPT_ROOT}/last.pt"
  fi

  # Evaluate on val and test sets
  python -m src.codonlm.eval_perplexity --ckpt "$EVAL_CKPT" --val_npz "$VAL_NPZ" 2>&1 | tee -a "$LOG" || true
  python -m src.codonlm.eval_perplexity --ckpt "$EVAL_CKPT" --val_npz "$TEST_NPZ" 2>&1 | tee -a "$LOG" || true

  # Score mutations for one CDS if primary DNA is available
  if [[ -n "$PRIMARY_DNA" ]]; then
    head -n1 "$PRIMARY_DNA" > data/processed/one_cds.txt
    mkdir -p "${SCORES_ROOT}"
    conda run -n codonlm python -m src.codonlm.score_mutations --ckpt "$EVAL_CKPT" --dna data/processed/one_cds.txt --out "${SCORES_ROOT}/one_cds__best.tsv" 2>&1 | tee -a "$LOG" || true
  else
    echo "[info] Bypassing mutation scoring: no primary DNA sequence available in pre-packed mode." | tee -a "$LOG"
  fi

  # Mine motifs (opt-in; skip if already present)
  if [[ $WITH_MOTIFS -eq 1 ]]; then
    if [[ -f "$RUN_DIR/motif_clusters.npz" ]]; then
      echo "[motifs] skip: already exists at $RUN_DIR/motif_clusters.npz" | tee -a "$LOG"
    else
      python -m src.codonlm.mine_motifs --ckpt "$EVAL_CKPT" --npz "$TRAIN_NPZ" --k 9 --clusters 100 2>&1 | tee -a "$LOG" || true
      if [ -f outputs/motif_clusters.npz ]; then
        cp outputs/motif_clusters.npz "$RUN_DIR/motif_clusters.npz" || true
      fi
    fi
  else
    echo "[motifs] skipped (enable with --with-motifs or run analysis.sh)" | tee -a "$LOG"
  fi

  # Collect artifacts (opt-in; skip if already present)
  if [[ $WITH_ARTIFACTS -eq 1 ]]; then
    if [[ -f "$RUN_DIR/artifacts.npz" ]]; then
      echo "[artifacts] skip: already exists at $RUN_DIR/artifacts.npz" | tee -a "$LOG"
    else
      python -m scripts.collect_artifacts_yaml "${RUN_ID}" "$CONF" 2>&1 | tee -a "$LOG" || true
    fi
  else
    echo "[artifacts] skipped (enable with --with-artifacts or run analysis.sh/post_process.sh)" | tee -a "$LOG"
  fi
fi

T1=$(date +%s)
TRAIN_SEC=$((Ttrain1-Ttrain0))
TOTAL_SEC=$((T1-T0))
echo "[timing] training_sec=${TRAIN_SEC}" | tee -a "$LOG"
echo "[timing] training_time=$(format_duration "$TRAIN_SEC")" | tee -a "$LOG"
echo "[timing] total_sec=${TOTAL_SEC}" | tee -a "$LOG"
echo "[timing] total_time=$(format_duration "$TOTAL_SEC")" | tee -a "$LOG"

if [[ "$TRAINER" == "codon_lm" && $DRY_RUN -eq 0 ]]; then
  echo "" | tee -a "$LOG"
  echo "[success] Training complete! To generate/design new sequences using this generator model, run:" | tee -a "$LOG"
  echo "python -m scripts.generative_design_loop --generator_ckpt $EVAL_CKPT" | tee -a "$LOG"
  echo "" | tee -a "$LOG"
fi
