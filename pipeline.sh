#!/usr/bin/env bash
# place data .gbff files in data/raw/

eval "$(conda shell.bash hook)"
conda activate codonlm

set -euo pipefail

usage() {
  cat >&2 <<USAGE
Usage: $0 [-c|--config PATH] [-r|--resume CHECKPOINT] [--dataset NAME,GBFF[,MIN_LEN]] [--force]
USAGE
  exit 1
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
      CONF="$2"
      shift 2
      ;;
    -r|--resume)
      [[ $# -lt 2 ]] && { echo "[error] --resume requires a path" >&2; usage; }
      RESUME="$2"
      shift 2
      ;;
    --dataset)
      [[ $# -lt 2 ]] && { echo "[error] --dataset requires NAME,GBFF[,MIN_LEN]" >&2; usage; }
      EXTRA_DATASETS+=("$2")
      shift 2
      ;;
    --force)
      FORCE=1
      shift
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
echo "[info] extra_datasets_cli=${EXTRA_DATASETS[*]:-none}" | tee -a "$LOG"

T0=$(date +%s)

# derive common hyperparameters from config (with defaults)
read BLOCK_SIZE WINDOWS_PER_SEQ VAL_FRAC TEST_FRAC <<<"$(python - <<'PY' "$CONF"
import yaml, sys
cfg = yaml.safe_load(open(sys.argv[1])) or {}
print(cfg.get("block_size", 256), cfg.get("windows_per_seq", 2), cfg.get("val_frac", 0.1), cfg.get("test_frac", 0.1))
PY
)"
COMBINED_DIR="data/processed/combined/${RUN_ID}"
MANIFEST_PATH="${RUN_DIR}/datasets_manifest.json"
mkdir -p "$COMBINED_DIR"

# compose dataset manifest from config + CLI extras
python - <<'PY' "$CONF" "$MANIFEST_PATH" "$BLOCK_SIZE" "$LOG" "$WINDOWS_PER_SEQ" "$VAL_FRAC" "$TEST_FRAC" "$FORCE" "${EXTRA_DATASETS[@]}"
import yaml, sys, json, pathlib
conf_path, manifest_path = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
block_size = int(sys.argv[3])
log_path = sys.argv[4]
windows_per_seq = float(sys.argv[5])
val_frac = float(sys.argv[6])
test_frac = float(sys.argv[7])
force = int(sys.argv[8])
extra_specs = sys.argv[9:]

cfg = yaml.safe_load(conf_path.open()) or {}
datasets = []

def add_entry(entry):
    name = entry["name"]
    gbff = entry["gbff"]
    if not pathlib.Path(gbff).exists():
        raise SystemExit(f"[error] GBFF not found: {gbff}")
    min_len = int(entry.get("min_len", 90))
    out_dir = pathlib.Path("data/processed") / name
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets.append({
        "name": name,
        "gbff": gbff,
        "min_len": min_len,
        "out_dir": str(out_dir),
        "dna": str(out_dir / "cds_dna.txt"),
        "meta": str(out_dir / "cds_meta.tsv"),
        "ids": str(out_dir / "codon_ids.txt"),
        "vocab": str(out_dir / "vocab_codon.txt"),
        "itos": str(out_dir / "itos.txt"),
        "train": str(out_dir / f"train_bs{block_size}.npz"),
        "val": str(out_dir / f"val_bs{block_size}.npz"),
        "test": str(out_dir / f"test_bs{block_size}.npz"),
    })

for entry in cfg.get("datasets", []):
    add_entry(entry)

for spec in extra_specs:
    parts = spec.split(",")
    if len(parts) < 2:
        raise SystemExit(f"[error] bad --dataset spec: {spec}")
    name, gbff = parts[0], parts[1]
    extra_entry = {"name": name, "gbff": gbff}
    if len(parts) > 2:
        extra_entry["min_len"] = int(parts[2])
    add_entry(extra_entry)

if not datasets:
    raise SystemExit("[error] No datasets specified in config or CLI.")

manifest = {
    "datasets": datasets,
    "block_size": block_size,
    "windows_per_seq": windows_per_seq,
    "val_frac": val_frac,
    "test_frac": test_frac,
    "force": force,
}
manifest_path.write_text(json.dumps(manifest, indent=2))
PY

# 1) Extract with meta
python - <<'PY' "$MANIFEST_PATH" "$LOG"
import json, subprocess, shlex, sys, os, pathlib
manifest = json.load(open(sys.argv[1]))
log = sys.argv[2]
datasets = manifest["datasets"]
block = int(manifest["block_size"])
windows = int(manifest["windows_per_seq"])
val_frac = float(manifest["val_frac"])
test_frac = float(manifest["test_frac"])
force = int(manifest["force"])

for ds in datasets:
    name = ds["name"]
    gbff = ds["gbff"]
    dna = ds["dna"]
    meta = ds["meta"]
    ids = ds["ids"]
    vocab = ds["vocab"]
    itos = ds["itos"]
    train = ds["train"]
    val = ds["val"]
    test = ds["test"]

    cmds = []
    if force or not (os.path.isfile(dna) and os.path.isfile(meta)):
        cmds.append([
            "python", "-m", "src.codonlm.extract_cds_from_genbank",
            "--gbff", gbff,
            "--out_txt", dna,
            "--out_meta", meta,
            "--min_len", str(ds["min_len"]),
        ])
    else:
        print(f"[skip] extract {name}")

    if force or not os.path.isfile(ids):
        cmds.append([
            "python", "-m", "src.codonlm.codon_tokenize",
            "--inp", dna,
            "--out_ids", ids,
            "--out_vocab", vocab,
            "--out_itos", itos,
        ])
    else:
        print(f"[skip] tokenize {name}")

    if force or not (os.path.isfile(train) and os.path.isfile(val) and os.path.isfile(test)):
        cmds.append([
            "python", "-m", "src.codonlm.build_dataset",
            "--ids", ids,
            "--group_meta", meta,
            "--block_size", str(block),
            "--windows_per_seq", str(windows),
            "--val_frac", str(val_frac),
            "--test_frac", str(test_frac),
            "--out_dir", str(pathlib.Path(train).parent),
        ])
    else:
        print(f"[skip] build {name}")

    for cmd in cmds:
        print(f"[run] {' '.join(shlex.quote(c) for c in cmd)}")
        subprocess.run(cmd, check=True)
PY 2>&1 | tee -a "$LOG"

COMBINED_MANIFEST=$(python - <<'PY' "$MANIFEST_PATH" "$COMBINED_DIR" "$BLOCK_SIZE"
import json, numpy as np, pathlib, sys
manifest=json.load(open(sys.argv[1]))
combined_dir=pathlib.Path(sys.argv[2])
block=int(sys.argv[3])
combined_dir.mkdir(parents=True, exist_ok=True)
datasets=manifest["datasets"]

def stack(paths):
    xs, ys = [], []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            xs.append(data["X"])
            ys.append(data["Y"])
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)

train_paths=[ds["train"] for ds in datasets]
val_paths=[ds["val"] for ds in datasets]
test_paths=[ds["test"] for ds in datasets]

train_out=combined_dir / f"train_bs{block}.npz"
val_out=combined_dir / f"val_bs{block}.npz"
test_out=combined_dir / f"test_bs{block}.npz"

X_train, Y_train = stack(train_paths)
np.savez_compressed(train_out, X=X_train, Y=Y_train)

X_val, Y_val = stack(val_paths)
np.savez_compressed(val_out, X=X_val, Y=Y_val)

X_test, Y_test = stack(test_paths)
np.savez_compressed(test_out, X=X_test, Y=Y_test)

combined_manifest = {
    "train": str(train_out),
    "val": str(val_out),
    "test": str(test_out),
    "datasets": datasets,
}
manifest_path = combined_dir / "manifest.json"
manifest_path.write_text(json.dumps(combined_manifest, indent=2))
print(str(manifest_path))
PY)

COMBINED_MANIFEST=${COMBINED_MANIFEST##*$'\n'}
cp "$COMBINED_MANIFEST" "$RUN_DIR/combined_manifest.json"
if [[ ! -f "$COMBINED_MANIFEST" ]]; then
  echo "[error] Failed to create combined manifest at $COMBINED_MANIFEST" >&2
  exit 1
fi
TRAIN_NPZ=$(python - <<'PY' "$COMBINED_MANIFEST"
import json, sys
data=json.load(open(sys.argv[1]))
print(data["train"])
PY)
VAL_NPZ=$(python - <<'PY' "$COMBINED_MANIFEST"
import json, sys
data=json.load(open(sys.argv[1]))
print(data["val"])
PY)
TEST_NPZ=$(python - <<'PY' "$COMBINED_MANIFEST"
import json, sys
data=json.load(open(sys.argv[1]))
print(data["test"])
PY)
PRIMARY_DNA=$(python - <<'PY' "$MANIFEST_PATH"
import json, sys
data=json.load(open(sys.argv[1]))
print(data["datasets"][0]["dna"])
PY)
echo "[info] combined_manifest=${COMBINED_MANIFEST}" | tee -a "$LOG"
echo "[info] train_npz=${TRAIN_NPZ}" | tee -a "$LOG"
echo "[info] val_npz=${VAL_NPZ}" | tee -a "$LOG"
echo "[info] test_npz=${TEST_NPZ}" | tee -a "$LOG"
echo "[info] primary_dna=${PRIMARY_DNA}" | tee -a "$LOG"

# 4) Train with MPS autocast, optional checkpointing, cosine LR
CKPT_ROOT="outputs/checkpoints/${RUN_ID}"
SCORES_ROOT="outputs/scores/${RUN_ID}"
Ttrain0=$(date +%s)
TRAIN_ARGS=(--config "$CONF" --run_id "${RUN_ID}" --train_npz "$TRAIN_NPZ" --val_npz "$VAL_NPZ" --test_npz "$TEST_NPZ")
if [[ -n "$RESUME" ]]; then
  TRAIN_ARGS+=(--resume "$RESUME")
fi
python -m src.codonlm.train_codon_lm "${TRAIN_ARGS[@]}" 2>&1 | tee -a "$LOG"
Ttrain1=$(date +%s)


# 5) Evaluate on val and test sets
python -m src.codonlm.eval_perplexity \
  --ckpt "${CKPT_ROOT}/best.pt" \
  --val_npz "$VAL_NPZ" 2>&1 | tee -a "$LOG" || true
python -m src.codonlm.eval_perplexity \
  --ckpt "${CKPT_ROOT}/best.pt" \
  --val_npz "$TEST_NPZ" 2>&1 | tee -a "$LOG" || true

# 6) Score mutations for one CDS (write to scores dir)
head -n1 "$PRIMARY_DNA" > data/processed/one_cds.txt
mkdir -p "${SCORES_ROOT}"
conda run -n codonlm python -m src.codonlm.score_mutations \
  --ckpt "${CKPT_ROOT}/best.pt" \
  --dna data/processed/one_cds.txt \
  --out "${SCORES_ROOT}/one_cds__best.tsv" 2>&1 | tee -a "$LOG" || true

# 7) Mine motifs (quick & dirty)
python -m src.codonlm.mine_motifs \
  --ckpt "${CKPT_ROOT}/best.pt" \
  --npz "$TRAIN_NPZ" --k 9 --clusters 100 2>&1 | tee -a "$LOG" || true
if [ -f outputs/motif_clusters.npz ]; then
  cp outputs/motif_clusters.npz "$RUN_DIR/motif_clusters.npz" || true
fi

# 8) Collect artifacts and organize run outputs under runs/<RUN_ID>
python -m scripts.collect_artifacts_yaml "${RUN_ID}" "$CONF" 2>&1 | tee -a "$LOG" || true

T1=$(date +%s)
echo "[timing] training_sec=$((Ttrain1-Ttrain0))" | tee -a "$LOG"
echo "[timing] total_sec=$((T1-T0))" | tee -a "$LOG"
