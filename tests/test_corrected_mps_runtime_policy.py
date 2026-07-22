import json
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_runtime_matrix_preserves_model_objective_and_effective_batch():
    base = yaml.safe_load((ROOT / "configs/corrected_mps_runtime_base.yaml").read_text())
    matrix = yaml.safe_load(
        (ROOT / "configs/corrected_mps_runtime_matrix.yaml").read_text()
    )

    assert base["dataset_manifest"].endswith(
        "corrected-codonlm-v1/genome/manifest.json"
    )
    assert (base["n_layer"], base["n_head"], base["n_embd"]) == (10, 8, 384)
    assert base["transfer_from"] is None
    assert base["device"] == "mps"

    permitted = {"batch_size", "grad_accum_steps", "use_checkpoint", "use_mmap"}
    for variant in matrix["variants"]:
        overrides = variant["overrides"]
        assert set(overrides) <= permitted
        assert overrides["batch_size"] * overrides["grad_accum_steps"] == 128


def test_recorded_runtime_policy_applies_predeclared_gate():
    evidence = json.loads(
        (ROOT / "docs/benchmarks/corrected_mps_runtime_gate.json").read_text()
    )
    acceptance = evidence["acceptance"]
    policy = evidence["selected_policy"]

    assert acceptance["mmap_throughput_ratio_vs_preloaded"] >= 0.95
    assert acceptance["mmap_dataset_rss_reduction_fraction"] >= 0.95
    assert acceptance["b4_no_checkpoint_speedup_vs_mmap_reference"] < acceptance[
        "minimum_candidate_speedup"
    ]
    assert acceptance["b8_no_checkpoint_speedup_vs_mmap_reference"] < 1.0
    assert acceptance["quality_gate_status"].startswith("not_entered")
    assert policy == {
        "batch_size": 4,
        "grad_accum_steps": 32,
        "use_checkpoint": True,
        "use_mmap": True,
        "amp": True,
        "n_kv_head": None,
        "use_sdpa": True,
        "sep_mask_enabled": True,
        "bucket_batching": False,
        "decision": "reference_runtime_retained_with_batch_aware_mmap",
    }


def test_recorded_ratios_match_raw_results():
    evidence = json.loads(
        (ROOT / "docs/benchmarks/corrected_mps_runtime_gate.json").read_text()
    )
    results = {row["name"]: row for row in evidence["results"]}
    reference = results["reference_mmap_b4_checkpoint"]
    preloaded = results["reference_preloaded_b4_checkpoint"]
    no_checkpoint = results["mmap_b4_no_checkpoint"]

    assert evidence["acceptance"]["mmap_throughput_ratio_vs_preloaded"] == pytest.approx(
        reference["non_pad_tokens_per_sec"] / preloaded["non_pad_tokens_per_sec"]
    )
    assert evidence["acceptance"][
        "b4_no_checkpoint_speedup_vs_mmap_reference"
    ] == pytest.approx(
        no_checkpoint["non_pad_tokens_per_sec"]
        / reference["non_pad_tokens_per_sec"]
    )
