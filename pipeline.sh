#!/usr/bin/env bash
# place data .gbff files in data/raw/

eval "$(conda shell.bash hook)"
conda activate codonlm

set -euo pipefail

usage() {
  echo "Usage: $0 [-c|--config PATH] [-r|--resume CHECKPOINT]" >&2
  exit 1
}

DEFAULT_CONF="configs/tiny_mps.yaml"
CONF="$DEFAULT_CONF"
RESUME=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--config)
      [[ $# -lt 2 ]] && { echo "[error] --config requires a path" >&2; usage; }
      CONF="$2"
      shift 2
      ;;
    -r|--resume)
      [[ $# -lt 2 ]] && { echo "[error] --resume requires a path" >&2; usage; }
      RESUME="$2"
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "[error] Unknown argument: $1" >&2
      usage
      ;;
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

# 0) Setup run id, log, and hardware info
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
import torch, platform, sys
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

T0=$(date +%s)

# 1) Extract with meta
python -m src.codonlm.extract_cds_from_genbank \
  --gbff data/raw/GCF_000005845.2_ASM584v2_genomic.gbff \
  --out_txt  data/processed/cds_dna.txt \
  --out_meta data/processed/cds_meta.tsv 2>&1 | tee -a "$LOG"

# 2) Tokenize to codon IDs (same as before)
python -m src.codonlm.codon_tokenize \
  --inp data/processed/cds_dna.txt \
  --out_ids data/processed/codon_ids.txt 2>&1 | tee -a "$LOG"

# 3) Build datasets with genome-aware train/val/test
python -m src.codonlm.build_dataset \
  --ids data/processed/codon_ids.txt \
  --group_meta data/processed/cds_meta.tsv \
  --block_size 256 --windows_per_seq 2 2>&1 | tee -a "$LOG"

# 4) Train with MPS autocast, optional checkpointing, cosine LR
CKPT_ROOT="outputs/checkpoints/${RUN_ID}"
SCORES_ROOT="outputs/scores/${RUN_ID}"
Ttrain0=$(date +%s)
TRAIN_ARGS=(--config "$CONF" --run_id "${RUN_ID}")
if [[ -n "$RESUME" ]]; then
  TRAIN_ARGS+=(--resume "$RESUME")
fi
python -m src.codonlm.train_codon_lm "${TRAIN_ARGS[@]}" 2>&1 | tee -a "$LOG"
Ttrain1=$(date +%s)


# 5) Evaluate on val and test sets
python -m src.codonlm.eval_perplexity \
  --ckpt "${CKPT_ROOT}/best.pt" \
  --val_npz data/processed/val_bs256.npz 2>&1 | tee -a "$LOG" || true
python -m src.codonlm.eval_perplexity \
  --ckpt "${CKPT_ROOT}/best.pt" \
  --val_npz data/processed/test_bs256.npz 2>&1 | tee -a "$LOG" || true

# 6) Score mutations for one CDS (write to scores dir)
head -n1 data/processed/cds_dna.txt > data/processed/one_cds.txt
mkdir -p "${SCORES_ROOT}"
conda run -n codonlm python -m src.codonlm.score_mutations \
  --ckpt "${CKPT_ROOT}/best.pt" \
  --dna data/processed/one_cds.txt \
  --out "${SCORES_ROOT}/one_cds__best.tsv" 2>&1 | tee -a "$LOG" || true

# 7) Mine motifs (quick & dirty)
python -m src.codonlm.mine_motifs \
  --ckpt "${CKPT_ROOT}/best.pt" \
  --npz data/processed/train_bs256.npz --k 9 --clusters 100 2>&1 | tee -a "$LOG" || true
if [ -f outputs/motif_clusters.npz ]; then
  cp outputs/motif_clusters.npz "$RUN_DIR/motif_clusters.npz" || true
fi

# 8) Collect artifacts and organize run outputs under runs/<RUN_ID>
python -m scripts.collect_artifacts_yaml "${RUN_ID}" "$CONF" 2>&1 | tee -a "$LOG" || true

T1=$(date +%s)
echo "[timing] training_sec=$((Ttrain1-Ttrain0))" | tee -a "$LOG"
echo "[timing] total_sec=$((T1-T0))" | tee -a "$LOG"
