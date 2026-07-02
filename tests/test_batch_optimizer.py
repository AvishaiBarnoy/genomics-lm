from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from scripts.optimize_train_batching import (
    DEFAULT_CANDIDATES,
    apply_remaining_wall_time_budget,
    benchmark_signature,
    include_current_candidate,
    load_cached_report,
    parse_candidates,
    resolve_optimizer_settings,
    select_best_result,
    train_command_for,
    write_report,
)


def test_parse_candidates_default_and_explicit():
    assert parse_candidates(None) == DEFAULT_CANDIDATES
    assert parse_candidates("2/16,4/32,8:16") == [(2, 16), (4, 32), (8, 16)]


def test_include_current_candidate_defaults_to_known_manual_pair():
    cfg = {"batch_size": 8, "grad_accum_steps": 32}
    candidates = [(2, 16), (4, 16), (8, 32)]

    assert include_current_candidate(cfg, candidates, include_current=True) == [
        (8, 32),
        (2, 16),
        (4, 16),
    ]


def test_resolve_settings_prefers_cli_over_config():
    cfg = {
        "force_gpu": False,
        "batch_optimizer": {
            "mode": "optimize",
            "candidates": [[2, 16]],
            "warmup_steps": 10,
            "measure_steps": 50,
            "force_gpu": False,
            "force": False,
            "include_current": True,
        },
        "batch_size": 2,
        "grad_accum_steps": 16,
    }
    args = argparse.Namespace(
        mode="benchmark",
        benchmark=False,
        optimize=False,
        candidates="4/16,8/32",
        warmup_steps=1,
        measure_steps=2,
        force_gpu=True,
        force=False,
    )

    settings = resolve_optimizer_settings(cfg, args)

    assert settings["mode"] == "benchmark"
    assert settings["candidates"] == [(2, 16), (4, 16), (8, 32)]
    assert settings["warmup_steps"] == 1
    assert settings["measure_steps"] == 2
    assert settings["force_gpu"] is True
    assert settings["force"] is False
    assert settings["include_current"] is True


def test_mode_flags_override_mode_and_config():
    cfg = {"batch_optimizer": {"mode": "benchmark"}}
    args = argparse.Namespace(
        mode="benchmark",
        benchmark=False,
        optimize=True,
        candidates=None,
        warmup_steps=None,
        measure_steps=None,
        force_gpu=False,
        force=False,
    )

    settings = resolve_optimizer_settings(cfg, args)

    assert settings["mode"] == "optimize"


def test_select_best_ignores_failed_and_tiebreaks():
    results = [
        {"status": "failed", "batch_size": 16, "grad_accum_steps": 16, "seq_per_sec": 999.0},
        {"status": "ok", "batch_size": 8, "grad_accum_steps": 32, "seq_per_sec": 10.0},
        {"status": "ok", "batch_size": 4, "grad_accum_steps": 16, "seq_per_sec": 10.0},
    ]

    selected = select_best_result(results)

    assert selected is not None
    assert selected["batch_size"] == 4
    assert selected["grad_accum_steps"] == 16


def test_write_report_outputs_files(tmp_path: Path):
    selected_cfg = {"batch_size": 4, "grad_accum_steps": 16, "force_gpu": True}
    selected = {
        "status": "ok",
        "batch_size": 4,
        "grad_accum_steps": 16,
        "seq_per_sec": 10.0,
    }
    results = [selected]
    command = train_command_for(tmp_path / "selected_config.yaml", "run-x", resume="runs/run-x/checkpoints/last.pt")

    write_report(tmp_path, results, selected, selected_cfg, command)

    assert (tmp_path / "results.csv").exists()
    payload = json.loads((tmp_path / "results.json").read_text())
    assert payload["selected"]["batch_size"] == 4
    cfg = yaml.safe_load((tmp_path / "selected_config.yaml").read_text())
    assert cfg["force_gpu"] is True
    assert "--resume" in (tmp_path / "train_command.txt").read_text()


def test_cached_report_requires_matching_signature(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("vocab_size: 10\nblock_size: 8\n")
    cfg = {"vocab_size": 10, "block_size": 8, "train_npz": "train.npz"}
    settings = {
        "candidates": [(2, 16)],
        "include_current": True,
        "warmup_steps": 1,
        "measure_steps": 2,
        "force_gpu": True,
    }
    signature = benchmark_signature(config_path, cfg, settings)
    selected = {"status": "ok", "batch_size": 2, "grad_accum_steps": 16, "seq_per_sec": 1.0}

    write_report(tmp_path, [selected], selected, {"batch_size": 2}, ["python"], signature=signature)

    cached = load_cached_report(tmp_path, signature)
    assert cached is not None
    results, cached_selected = cached
    assert results[0]["batch_size"] == 2
    assert cached_selected["grad_accum_steps"] == 16

    changed = dict(signature)
    changed["measure_steps"] = 3
    assert load_cached_report(tmp_path, changed) is None


def test_cached_report_accepts_legacy_config_sha_signature(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("vocab_size: 10\nblock_size: 8\n")
    cfg = {"vocab_size": 10, "block_size": 8, "train_npz": "train.npz"}
    settings = {
        "candidates": [(2, 16)],
        "include_current": True,
        "warmup_steps": 1,
        "measure_steps": 2,
        "force_gpu": True,
        "include_in_wall_time": True,
        "min_training_minutes_after_opt": 0.0,
    }
    signature = benchmark_signature(config_path, cfg, settings)
    legacy_signature = dict(signature)
    legacy_signature["config_sha256"] = "old-full-file-hash"
    selected = {"status": "ok", "batch_size": 2, "grad_accum_steps": 16, "seq_per_sec": 1.0}

    write_report(tmp_path, [selected], selected, {"batch_size": 2}, ["python"], signature=legacy_signature)

    assert load_cached_report(tmp_path, signature) is not None


def test_apply_remaining_wall_time_budget_reduces_training_budget():
    cfg = {"max_time_minutes": 60, "batch_size": 4}

    selected_cfg, remaining = apply_remaining_wall_time_budget(
        cfg,
        elapsed_seconds=900,
        include_in_wall_time=True,
    )

    assert remaining == 45
    assert selected_cfg["max_time_minutes"] == 45
    assert selected_cfg["batch_optimizer_original_max_time_minutes"] == 60
    assert selected_cfg["batch_optimizer_elapsed_minutes"] == 15
    assert cfg["max_time_minutes"] == 60


def test_apply_remaining_wall_time_budget_can_be_disabled():
    cfg = {"max_time_minutes": 60}

    selected_cfg, remaining = apply_remaining_wall_time_budget(
        cfg,
        elapsed_seconds=900,
        include_in_wall_time=False,
    )

    assert remaining is None
    assert selected_cfg["max_time_minutes"] == 60
