#!/usr/bin/env bash

RUN_ID=$1
if [ -z "$RUN_ID" ]; then
    echo "Error: no RUN_ID provided."
    echo "Usage: $0 <RUN_ID>" >&2
    exit 1
fi

CONF=${CONF:-configs/tiny_mps_v2.yaml}

python -m scripts.collect_artifacts_yaml "$RUN_ID" "$CONF"
python -m scripts.analyze_frequencies   "$RUN_ID"
python -m scripts.analyze_embeddings    "$RUN_ID"
python -m scripts.analyze_attention     "$RUN_ID"
python -m scripts.probe_next_token      "$RUN_ID"
python -m scripts.analyze_saliency      "$RUN_ID"
python -m scripts.report_top_saliency   "$RUN_ID" --window 9 --top 20

# Ensure labels exist before probing
python -m scripts.generate_probe_labels "$RUN_ID"
python -m scripts.probe_linear          "$RUN_ID"

# If mutation scores exist under outputs/, copy into run dir so the summarizer can find them
RUN_DIR="runs/$RUN_ID"
SCORES_FILE="outputs/scores/$RUN_ID/one_cds__best.tsv"
if [ -f "$SCORES_FILE" ]; then
    mkdir -p "$RUN_DIR"
    cp "$SCORES_FILE" "$RUN_DIR/one_cds__best.tsv"
fi

python -m scripts.summarize_one_cds     "$RUN_ID"  # optional
# python -m scripts.compare_runs "$RUN_ID" <other_run_ids...>
