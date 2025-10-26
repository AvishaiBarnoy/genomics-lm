#!bash
RUN_ID=2025-09-30_tiny_2L4H_d128_e5
RUN_ID=$1
PATH_TO_CONF=./configs/
python -m scripts.collect_artifacts_yaml $RUN_ID $PATH_TO_CONF/tiny_mps.yaml
python -m scripts.analyze_frequencies   $RUN_ID
python -m scripts.analyze_embeddings    $RUN_ID
python -m scripts.analyze_attention     $RUN_ID
python -m scripts.probe_next_token      $RUN_ID
python -m scripts.analyze_saliency      $RUN_ID
python -m scripts.probe_linear          $RUN_ID
python -m scripts.summarize_one_cds     $RUN_ID  # optional
#python -m scripts.compare_runs $RUN_ID <other_run_ids...>
