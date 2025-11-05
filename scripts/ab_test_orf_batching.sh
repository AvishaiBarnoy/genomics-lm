#!/usr/bin/env bash
# Run A/B experiment: CDS+SEP (multi, masked) vs single-CDS (no SEP mask) from the same base config.
# Usage: ./scripts/ab_test_orf_batching.sh -c configs/tiny_mps.yaml

set -euo pipefail

CONF=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--config) CONF="$2"; shift 2;;
    *) echo "Unknown arg: $1" >&2; exit 1;;
  esac
done

if [[ -z "$CONF" ]]; then
  echo "Usage: $0 -c CONFIG.yaml" >&2
  exit 1
fi

if [[ ! -f "$CONF" ]]; then
  echo "Config not found: $CONF" >&2
  exit 1
fi

BASE=$(basename "$CONF")
DIR=$(cd "$(dirname "$CONF")" && pwd)
TMP_A=$(mktemp -t abA.XXXX.yaml)
TMP_B=$(mktemp -t abB.XXXX.yaml)

python - "$CONF" "$TMP_A" <<'PY'
import sys, yaml
cfg=yaml.safe_load(open(sys.argv[1])) or {}
cfg['pack_mode']='multi'
cfg['sep_mask_enabled']=True
yaml.safe_dump(cfg, open(sys.argv[2],'w'))
PY

python - "$CONF" "$TMP_B" <<'PY'
import sys, yaml
cfg=yaml.safe_load(open(sys.argv[1])) or {}
cfg['pack_mode']='single'
cfg['sep_mask_enabled']=False
yaml.safe_dump(cfg, open(sys.argv[2],'w'))
PY

echo "[A/B] Running A (multi+masked)"
RUN_ID="$(python -m scripts.make_run_id "$TMP_A")-A"
RUN_ID="$RUN_ID" ./main.sh -c "$TMP_A"

echo "[A/B] Running B (single+unmasked)"
RUN_ID="$(python -m scripts.make_run_id "$TMP_B")-B"
RUN_ID="$RUN_ID" ./main.sh -c "$TMP_B"

echo "[A/B] Compare runs"
python -m scripts.compare_runs

